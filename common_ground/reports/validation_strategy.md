# Validation strategy — Common Ground (NYVEST)

**Date:** 2026-05-19
**Scripts:**
[llto_cg_validation.py](../scripts/llto_cg_validation.py) ·
[scripts/extraction/sample_feature_space_stable_allyears.py](../../scripts/extraction/sample_feature_space_stable_allyears.py)

This document outlines the validation harness used for the Common
Ground semi-supervised pipeline. The plain-English summary first, then
the mechanical detail.

> Earlier transfer-eval and naive-anchor harnesses were retired on
> 2026-05-19 because they graded against a Stage 1@2020 anchor instead
> of grunnkart labels on held-out stable points. See git history for
> the deleted scripts and reports if a reference is needed.

---

## 1. Inputs and what each one represents

| Dataset | File | Role |
|---|---|---|
| **Stable parquet** | `data/grunnkart_nyvest_fscs_alphaearth.parquet` | t₀ trusted-label set. AlphaEarth (64 bands) at CCDC-unchanged pixels, year=2020 only. Grunnkart class is treated as ground truth. |
| **Unstable parquet** | `data/grunnkart_nyvest_fscs_unstable_alphaearth.parquet` | t₁ change-affected set. 64 bands per year for the same 1843 points across years 2017–2025, sampled at CCDC-broken pixels. Grunnkart label here is decayed → needs pseudo-labelling. |

The two parquets are disjoint at the pixel level by construction
(stable mask vs unstable mask), so the same point never appears in
both.

---

## 2. What we are validating

**Question.** Does the Common Ground SSL pipeline (Stage 1 on stable →
pseudo-label unstable → Stage 2 on stable + pseudo-labelled unstable)
improve classification at held-out stable points, with and without a
conformal filter on the pseudo-labels?

**Harness.** [llto_cg_validation.py](../scripts/llto_cg_validation.py)
implements **Leave-Location-Time-Out (LLTO)** CV where folds are spatial
clusters built from the stable parquet. For each fold k it trains Stage
1 + Stage 2 on all data outside the fold, then scores Stage 2 against
the grunnkart label on the held-out stable rows of fold k.

The companion **naive arm** — Stage 1 trained on non-OOF data and
applied directly to OOF stable points across all years — is deferred
until the multi-year stable parquet exists (see §6).

---

## 3. Leave-Location-Time-Out (LLTO) folds — definition and rationale

Spatial folds are built via **KMeans(k=n_folds) on (lon, lat) of the
stable parquet**. Each stable point is assigned a fold id from
`fit_predict`. Unstable points inherit the same fold via
`km.predict(unstable[["lon","lat"]])` — so a fold = a spatial region
across both parquets, and same-location points always share a fold.

- **Same physical location ⇒ same fold in every era.** This prevents
  *background memorisation* — the model recognising the surroundings
  of a test pixel because it saw a snapshot of the same area during
  training.
- **CV is spatial, not random.** Random splits inflate F1 by 0.05–0.10
  in this domain because adjacent 10 m AlphaEarth pixels are
  near-identical; LLTO removes that easy leakage.
- **Folds may be imbalanced.** KMeans yields geographically compact,
  not equal-sized, folds. We report `n_train` / `n_test` per fold so
  the reader sees the heterogeneity. With k=3 the stable fold sizes
  are ~25 k / 23 k / 32 k and unstable inheriting them lands at
  2 799 / 369 / 13 676 — the small unstable fold is a known sparsity
  artefact of KMeans on this geometry.

---

## 4. Per-fold pipeline

For fold k:

- `train_stable`   = stable rows with fold ≠ k (year=2020 only).
- `train_unstable` = unstable rows with fold ≠ k (all years 2017–2025).
- `test_stable`    = stable rows with fold == k (year=2020).

Inside `train_stable`, a frozen, **stratified 20% slice** is held out
as `raps_cal` for conformal calibration; the remaining 80% trains
Stage 1. The cal slice is built fresh per fold so it never overlaps
the test fold.

### Two setups per fold
- `cg_llto`        — Stage 2 = `train_stable + Stage1.predict(train_unstable)`
                     (no filter).
- `cg_llto_raps`   — same as `cg_llto` but pseudo-labels are gated by
                     a RAPS singleton check (α=0.10, λ=0.01, k_reg=1,
                     τ calibrated on `raps_cal`).

### Test
Stage 2 predicts `test_stable` and is graded against the **grunnkart
class**. Metrics: macro-F1, balanced accuracy, per-class F1, per-fold
and aggregated confusion matrices in
[llto_cg/confusion/](llto_cg/confusion/).

---

## 5. RAPS singleton filter

The repo's [conformal_compare.md](../../reports/conformal_compare.md)
benchmarked APS / RAPS / SAPS / RANK on a similar tabular AlphaEarth
setup. For automated downstream pipelines it recommends **RAPS at
α=0.10 with λ=0.01, k_reg=1**:

> Smaller sets, no empties, better statistical efficiency.

RAPS score for class k on sample x:
`V(p, k) = Σ_{j<rank(k)} p_(j) + u · p_(k) + λ · max(0, rank(k) − k_reg)`.

We calibrate τ on `raps_cal`, then build a RAPS set for every
pseudo-labelled unstable row. A **singleton set** is the conformal
analogue of "the model is confident in exactly one class at coverage
1 − α." Keeping only those rows for Stage 2 trades pseudo-label
quantity for label quality, with a calibrated guarantee rather than a
hand-picked probability threshold.

Coverage and singleton diagnostics from the cal slice are logged
(`raps_tau`, `cal_singleton`, `top1_acc_cal`) so we can sanity-check
that the calibration held on stable data before the filter is applied
to the unstable rows. Note that exchangeability between `raps_cal` and
the unstable rows is **only approximate** — the unstable rows live on
CCDC-broken pixels and across years 2017–2025. The conformal guarantee
is exact on stable; on unstable we treat it as a strong heuristic, not
a proof.

---

## 6. Deferred — naive arm

A like-for-like naive comparison would predict OOF stable points across
all years 2017–2025 (the equivalent of CG's per-year unstable pass but
applied to stable pixels). Stable parquet today is year=2020 only, so
the extraction script
[scripts/extraction/sample_feature_space_stable_allyears.py](../../scripts/extraction/sample_feature_space_stable_allyears.py)
mirrors the unstable sampler with the *stable* CCDC mask kept in its
original sense. Run it on GEE and `llto_naive_validation.py` (to be
written) will slot in alongside the CG redux.

---

## 7. Results (3 folds, CatBoost iter=500, α=0.10)

| setup | macro-F1 mean ± std | bal_acc mean ± std | RAPS kept % |
|---|---|---|---|
| cg_llto | 0.620 ± 0.028 | 0.612 ± 0.029 | — |
| cg_llto_raps | 0.616 ± 0.032 | 0.607 ± 0.036 | 30.4% |

CG and CG+RAPS are statistically a tie at fold-level (Δ = −0.003 F1, σ
≈ 0.030). The filter has very little net effect at the macro level on
this LLTO test — class-level deltas are in
[llto_cg/llto_cg_per_class_f1.csv](llto_cg/llto_cg_per_class_f1.csv)
and the mean pivot in
[llto_cg/llto_cg_per_class_f1_mean.csv](llto_cg/llto_cg_per_class_f1_mean.csv).

---

## 8. Evaluation metrics

For every fold and setup we report:

- **macro-F1** — equal weight per class, robust to the heavy class
  imbalance in grunnkart (class 12 has only 159 rows; class 6 has
  12 771).
- **balanced accuracy** — recall averaged across classes; companion to
  macro-F1.
- **raps_kept_pct** — fraction of pseudo-labels passing the RAPS
  singleton filter (only meaningful for `cg_llto_raps`).
- Per-class F1 + 12 × 12 confusion matrix per fold and aggregated.

Per-fold rows live in
[llto_cg/llto_cg_per_fold.csv](llto_cg/llto_cg_per_fold.csv); summaries
in [llto_cg/llto_cg_summary.csv](llto_cg/llto_cg_summary.csv).

---

## 9. Known caveats

1. **Fold imbalance.** KMeans gives compact clusters, not equal-sized
   ones. Fold 1 has only 369 unstable training rows; that fold's
   Stage 2 is essentially the stable-only baseline with a tiny pseudo
   nudge. Per-fold variance is wide; the mean is the more reliable
   summary.
2. **Approximate exchangeability** between `raps_cal` (stable) and the
   unstable pseudo-labelled rows (see §5). The RAPS guarantee is exact
   on stable; treat it as a strong heuristic on unstable.
3. **No filter on the t₀ side.** Stage 1 is trained on every
   `stable_train` row; we don't drop low-confidence stable rows because
   they carry trusted labels, not pseudo-labels.
4. **Single test year.** Test points are only graded at year=2020,
   because that is the only AlphaEarth slice the stable parquet
   contains. Multi-year naive vs CG comparisons are blocked on the
   §6 extraction.
