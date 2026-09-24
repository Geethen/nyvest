# ar3 — can the DEPLOYED moe8 still be improved?

Started 2026-09-23. Successor to `DNN/autoresearch/` (71 mechanisms, zero wins).
Everything there was scored on ONE metric (macro-F1) on ONE population (the FSCS
stable points). This round changes both, because the two ideas under test can
only show up off that population.

## Baseline = the deployed model

`models/dnn_final_moe8_merged.pt`: moe_shared, 8 routed experts top-2 + shared
256,128 trunk, 9 merged classes (`MERGE_EXTRA=11:2`), lidar, `to12_fix`,
5-member ensemble. CV reference macro-F1 0.7542, OOF ECE 0.051 raw / 0.006
Venn-Abers, conformal (α=0.1) mean set 1.47, singleton 60%.

## Why the old metric cannot see the two ideas

The training AND test points are drawn from CCDC-stable pixels, one per k-means
cluster per (25 km cell, class). That frame systematically excludes the pixels where
the deployed maps flip between 2018 and 2024 (7.1% of the AOI; 21% of 2018-built
"leaves" built). So:

* **consensus labels in error-prone areas** can help exactly where the old
  test set never looks → need a HARD-AREA test population.
* **temporal information** mainly buys consistency across years → need a
  temporal-consistency metric. Free proxy: each FSCS test point has 9 year
  rows under one label, CCDC-stable over (2017, 2020]. Any year-to-year flip
  in its prediction is an error.

## Metric suite (every trial, paired per fold vs a fresh baseline run)

| family | metric | population |
|---|---|---|
| accuracy | macro-F1, per-class F1, OA | FSCS test fold (legacy, must not regress) |
| accuracy | macro-F1, OA | consensus hard-area points in test cells |
| calibration | ECE (15-bin top-label), NLL, Brier; + after group-val temp scaling | FSCS test fold |
| confidence | split-conformal LAC α=0.1 fitted on group-val: coverage, mean set size, singleton rate | FSCS test fold |
| temporal | flip rate 2017-2020 consecutive pairs; 2018↔2024 flip rate | FSCS test points |

Noise floors: baseline run at TWO seed sets (AR_SEED 0 and 1) and the per-fold
sd of every metric is recorded. A metric delta is only "real" if all folds agree
and |Δmean| > 2× that metric's seed sd. The 0.0013 macro-F1 floor from the
last round is the precedent.

Verdict per trial = a scorecard, not one number: `improves` (≥1 metric real
improvement), `regresses` (any metric real regression), `tie`. The model to
adopt must improve something and regress nothing. The top result gets
seed-replicated before it is believed (winner's curse, see last round).

## Workstreams

**W1 — consensus labels (delegated: copilot CLI / Sonnet 5)** →
`briefs/phase1_consensus.md`. Sample error-prone pixels from the 2018/2024 maps
(flip / uncertain / random-control strata, spread across 25 km cells), then
extract grunnkart v1/v2, ESA WorldCover 2020/2021, Esri 10 m LULC, Dynamic World
and AEF 2017-2025 + lidar at those points. Raw product labels are stored, and
the consensus RULE is applied in the harness so that trials can vary it.

**W2 — harness + metrics (me)** → `ar3_common.py`, `metrics.py`.

**W3 — trials (delegated: Claude Sonnet subagent, reviewed by me)** → `trials3.py`.
Pre-registered queue:

temporal (no new data):
- `tmp_consist`: train-time KL consistency between the same location's
  predictions in different years (single-year inference, zero deploy cost).
- `tmp_pair`: input = [emb_y, emb_other] for the 2018/2024 pair (deployable with the existing VRTs).
- `tmp_ctx`: input = [emb_y, mean_y', std_y'] over all 9 years (deploy cost: 9 VRTs, flagged).
- `tmp_joint`: post-hoc bi-temporal joint decoding, P(c18,c24) ∝ p18·p24·T,
  sticky T with ε ∈ {0.002, 0.01, 0.05} pre-registered (no fitting on stable
  points, where the optimum is trivially ε→0).
- `tmp_pool` (bound): average the posterior over all 9 years. It is the
  ceiling for temporal denoising on stable points.

calibration / confidence:
- `cal_ls0`: label smoothing 0 (LS distorts calibration).
- `cal_focal`: focal loss γ=3 (Mukhoti et al. 2020, calibration).
- `cal_ts_gval`: temperature fitted on group-val (spatial transfer) vs random val.
- `cal_mondrian`: class-conditional conformal vs LAC set size.

consensus data (after W1):
- `cons_add`: + consensus-labelled hard-area rows in TRAIN cells.
- `cons_add_w05`: same, weight 0.5.
- `ctrl_hard_gk`: + the SAME hard-area points with raw grunnkart labels, no
  consensus filter (separates "labels in hard areas" from "the filter").
- `ctrl_rand_cons`: + consensus rows from the RANDOM stratum (separates
  "hard areas" from "more consensus data").
- `cons_combo`: best consensus arm + best temporal arm.

**W4 — loop** (`loop3.py`): screen at N_ENSEMBLE=3, confirm the top arms at
N_ENSEMBLE=5 + AR_SEED=1.

## Priors from the last round (do not re-learn)

More data of the same kind has not helped here: NiN −, v2-built −0.0095 on built,
scaling-data flat. Consensus selection is likely to pick the EASY pixels inside
hard areas, where all products agree because the pixel is spectrally obvious.
`ctrl_hard_gk` and the hard-area test metric exist to catch that.
