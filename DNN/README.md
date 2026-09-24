# DNN replacement for the CatBoost + TabICL pipeline

A GPU DNN classifier for the nyvest land-cover task, built to replace the
two-stage CC-APS CatBoost → TabICL model. Evaluated on the **exact same
protocol** as `common_ground/scripts/llto_schemeB_allyears.py` so macro-F1 is
directly comparable:

- stable-allyears parquet (663,740 rows, 12 classes)
- class merge 1→2 and 9→8 (⇒ 10 effective classes)
- 3-fold spatial CV via `GroupKFold` on `cell_id`
- macro-F1 over all classes (`zero_division=0`)
- 64 AlphaEarth bands + 3 lidar features (elevation, tri, tch)
- test-fold labels never cleaned/relabelled (leak-free)

**Reference to beat: TabICL+lidar macro-F1 = 0.7139.**

**The floor, for scale** (same folds/weights/features, `autoresearch/trials_classic.py`):
a linear probe on the 67 features scores **0.7040** and a tuned random forest
**0.6945**. The deployed net's whole margin over a linear read is **+0.0373**, and
three rounds of mechanism search have been contesting **+0.0040** of that.

## Results (3-fold spatial CV macro-F1)

| Stage | Model | F1 mean | std | vs 0.7139 | wall time |
|------|-------|--------:|----:|----------:|----------:|
| 1 | plain 2-layer MLP, class-weighted CE | 0.7112 | 0.009 | −0.0027 | ~60 s |
| 2 | residual + BatchNorm MLP, SWA, ensemble | 0.7106 | 0.007 | −0.0033 | ~610 s |
| 3 | robust MLP (mixup + input noise + 5-seed ensemble + cls12 clean) | 0.7283 | 0.009 | +0.0144 | ~430 s |
| 3 (final) | MLP + 5-seed ensemble + cls12 clean (mixup/noise off) | 0.7318 | 0.010 | +0.0179 | ~270 s |
| **8 (best)** | **+ targeted cls12-only relabel (to12_fix)** | **0.7341** | 0.009 | **+0.0202** | ~370 s |

**Key finding 1 — capacity is not the lever.** Stage 2's residual net drove
val-F1 to 0.965 while test-F1 stayed at 0.71: depth/BatchNorm memorizes the train
fold and does not transfer across spatial (cell_id) folds. The bottleneck is
*spatial generalization*, not expressiveness.

**Key finding 2 — ensembling is the lever; augmentation is not.** The Stage-3
ablation (drop one lever at a time from the 0.7283 config):

| variant | F1 | Δ vs 0.7283 |
|---|---:|---:|
| no mixup | 0.7289 | +0.0006 |
| no input-noise | 0.7297 | +0.0014 |
| no cls12 clean | 0.7279 | −0.0004 |
| single seed (no ensemble) | 0.7248 | −0.0035 |
| **no mixup + no noise** | **0.7318** | **+0.0035** |

mixup and input-noise slightly *hurt* here — dropping both is best. The reliable
gain is the **5-seed probability ensemble** (single-seed loses 0.0035). cls12
centroid cleaning is macro-F1-neutral but kept for label correctness and
leak-safety. The final/default config is therefore the simplest one: a 2-layer
class-weighted MLP, 5-seed ensembled, with train-only cls12 cleaning.

## Stage 4 — conformal (CC-APS) pseudo-labelling from the unstable parquet

Self-training: per fold, train the DNN ensemble on stable-fit, calibrate an APS
gate on a held-out stable-cal slice (α=0.05, same as the reference), pseudo-label
the unstable rows whose APS set is a **singleton**, then retrain on stable +
pseudo. Leak-free (test fold never touched).

**Result: F1 = 0.7320 ± 0.009 — a tie with 0.7318 (+0.0002, within fold noise).**
The gate works (keeps ~31% of unstable rows at ~89% pseudo-label accuracy), but
adding ~5k pseudo rows to an already-664k-row training set is a rounding error.
This is the opposite of TabICL, where support is capped at 32k so each added row
counts. The unstable set is also small (16.8k rows) and has **no class 12**, so
it can't help the hardest class. **Verdict: no-op for the DNN; not adopted.**

## Architecture sweep (Stage 5) and MoE (Stage 6) — all lose to the plain MLP

Same robust recipe (5-seed ensemble, sqrt weights, label smoothing, cls12
clean), only the architecture changes:

| Architecture | F1 mean | vs MLP best (0.7318) |
|---|---:|---:|
| **plain MLP** | **0.7318** | — |
| GLU / GEGLU gated MLP | 0.7288 | −0.0030 |
| MoE, 4 soft experts (stage6) | 0.7294 | −0.0024 |
| residual GLU | 0.7206 | −0.0112 |
| SNN (SELU + AlphaDropout) | 0.7075 | −0.0243 |

More expressive nets overfit the spatial folds (residual-GLU val-F1 0.88,
test 0.72) — same lesson as Stage 2. The MoE did **not** specialize its way to
better grassland/sparse-veg; problem classes were flat-to-worse. Capacity is not
the lever.

## Spatial MoE (Stage 7) — experts specialized by GEOGRAPHY also lose

`stage7_spatial_moe.py`: experts route by location, not class. Two modes —
`soft` (gating MLP on lon/lat → softmax) and `hard` (KMeans regions on train
coords, leak-free, one expert per region). Same robust recipe otherwise.

| Config | F1 mean | std | vs MLP best (0.7318) | gate behavior |
|---|---:|---:|---:|---|
| **plain MLP** | **0.7318** | .010 | — | — |
| class-MoE E=4 | 0.7294 | .008 | −0.0024 | — |
| soft spatial E=4 | 0.7280 | .011 | −0.0038 | 3-4 experts shared |
| soft spatial E=8 | 0.7221 | .008 | −0.0097 | ~5 shared |
| hard KMeans E=4 | 0.7166 | .012 | −0.0152 | ~1 region/fold |
| hard KMeans E=8 | 0.7092 | .005 | −0.0226 | ~1 region/fold (worst) |

**All lose; soft > hard; more experts = worse.** The gate-usage diagnostic
explains hard's collapse: each spatial-CV test fold (GroupKFold on cell_id) falls
almost entirely inside ONE KMeans region, so the gate routes ~84-95% of the fold
to a single expert (e.g. E=8 fold0→expert0 @0.839, fold1→expert1 @0.952). The
other experts never see the test region and are dead weight, leaving one
under-trained expert to predict the whole fold. **Hard spatial experts are
redundant with, and antagonistic to, the geographic CV.** Soft routing blends and
degrades gracefully but still can't beat a single model that already sees all
regions. Confirms again: the bottleneck is feature-space class overlap, not a
routing/capacity problem — neither class-experts nor region-experts help.

## Locality round (autoresearch round 3) — modern-LLM MoE ties instead of losing

`DNN/autoresearch/{moe_layers,trials_local,loop_local,diag_locality}.py`, live
page + W&B as in the rest of `autoresearch/`. Stage 6/7 diagnosed MoE failure as
routing collapse (hard KMeans put 84-95% of a held-out fold on one expert), so
this round rebuilt the idea around that diagnosis, using the MoE design the LLM
world converged on: **an always-on shared expert plus small routed experts**
(DeepSeekMoE), with the routed branch zero-initialised so the net *starts* at the
deployed model and can only add to it. Gate source is the experiment — `content`
(the row's own features, what an LLM routes on), `terrain` (elevation/tri/tch),
and `geo` (lon/lat) as the paired stage7-style control.

| Arm | gate reads | params | F1 | Δ paired | inner val |
|---|---|---:|---:|---:|---:|
| baseline | — | 51.6k | 0.7414 | — | 0.7974 |
| moe_shared | content | 78.9k | 0.7437 | +0.0023 (all folds +) | 0.8152 |
| **ctrl_capacity** | *no router* | 78.0k | 0.7421 | +0.0007 | **0.8121** |
| **mod_depth** | *adaptive depth* | 84.9k | 0.7441 | **+0.0028** (all folds +) | **0.8040** |
| moe_geo | lon/lat | 78.7k | 0.7428 | +0.0014 | 0.8146 |
| moe_lossfree | content (bal.) | 78.9k | 0.7434 | +0.0020 | — |
| moe_lora | content (LoRA) | 78.9k | 0.7416 | +0.0003 | — |
| moe_echoice | content (expert-choice) | 78.9k | 0.7439 | +0.0026 | 0.8118 |
| moe_auxbal | content (Switch+z) | 78.9k | 0.7427 | +0.0013 | 0.8142 |
| moe_terrain | elev/tri/tch | 78.7k | 0.7414 | +0.0001 | 0.8110 |
| film_hyper | terrain (FiLM) | 101.8k | 0.7397 | −0.0016 | 0.8283 |
| moe_soup | *regional merge* | 51.6k | 0.7396 | −0.0017 | — |
| knn_retrieval | — (kNN-LM) | 51.6k | 0.7384 | +0.0012 *vs group-val control* | 0.7455 |

**Eleven mechanisms, no wins.** Two families worth naming: the four load-balancing
variants (none / loss-free bias / Switch+z-loss / expert-choice) span +0.0013 to
+0.0026, so *how* the router is balanced matters as little as *what it reads*; and
`moe_soup` — fine-tune one copy per KMeans region, average in weight space, the
most literal reading of "local models" — is the second-worst arm at −0.0017. Its
pre-registered control (`n_regions=1`: same extra fine-tuning, nothing regional)
scores −0.0008, so extra fine-tuning costs 0.0008 and splitting it geographically
costs a further 0.0009 — **the specifically *local* half is the larger harm.**

**Ranked by inner val, the held-out delta runs the other way.** The two arms that
fit the training regions best (`moe_lora` val 0.8418, `film_hyper` 0.8283) land at
+0.0003 and −0.0016; the arm with the *lowest* val (`mod_depth` 0.8040) is the only
one clearing +0.0025. Pearson r(val, Δtest) = −0.60 (p=0.07), Spearman −0.46
(p=0.19) — a negative trend that does **not** reach significance at n=10 and leans
on those two arms, so treat it as suggestive. It points the same way as the
capacity control, the scaling study and the reg sweep.

**Five results worth keeping:**

0. **`ctrl_capacity` is the one that matters — run it before believing any
   val-side gain here.** A plain MLP widened to the MoE's parameter count reaches
   val 0.8121, and on fold 1 it *beats* the MoE's val outright (0.8104 vs 0.8089).
   Of the MoE's +0.0178 val gain, **capacity accounts for +0.0147 and routing for
   +0.0031**; on the held-out region the split is +0.0007 / +0.0016. Routing's
   share of both is inside the noise floor. So the result is *not* "locality is
   learned in-distribution and lost at the fold boundary" — a routed local branch
   does about as much as the same weights added in a straight line.

1. **Routing collapse was the wrong diagnosis.** Under soft top-2 routing the
   busiest expert's share on the *held-out* region matches its share on the
   *training* rows to within 0.01 on every arm and fold (content .426/.435,
   terrain .524/.527, lon/lat .305/.310). The partition transfers intact; the
   knowledge attached to it does not. What separates these ties from stage7's
   −0.004…−0.023 losses is **shared-expert isolation** — keeping the global model
   rather than replacing it with a partition — not the routing signal.
2. **Gate source doesn't matter.** Content / position / terrain span +0.0023 to
   +0.0001, narrower than the harness noise floor (~0.0013). The round was
   designed to show content ≫ position; it does not.
3. **Adaptive DEPTH is the one thing the capacity control does not explain — the
   only follow-up worth having.** `mod_depth` (Raposo et al. 2024) routes the
   hardest 50% of rows through an extra zero-initialised residual block and lets
   the rest bypass; an auxiliary predictor reproduces the top-k decision at
   inference (held-out deep-fraction 0.50–0.53, matching the 0.5 target). It has
   *more* parameters than `ctrl_capacity` yet scores *lower* on inner val (0.8040
   vs 0.8121) while scoring *higher* on the held-out region — the reverse of the
   capacity signature, i.e. generalisation rather than memorisation. All four weak
   classes up, worst strong class −0.0013. It halved to +0.0013 on a fresh seed
   (winner's curse) but stayed positive on **all 6 fold×seed combinations**,
   averaging +0.0021. The pre-registered capacity sweep gives a **symmetric
   interior optimum** — 0.25 → +0.0017, **0.50 → +0.0028**, 0.75 → +0.0017 — with
   both ends collapsing toward things already measured (at 0.75 nearly every row
   takes the block = plain capacity, +0.0007; at 0.25 almost none does = baseline).
   The peak sits exactly where the routing decision is selective. Margins are near
   the 0.0013 noise floor, so **the shape is more informative than the numbers**.
   Still a tie against the +0.003 bar. **Next step: a 5-seed average at capacity
   0.5 rather than more single draws.**
4. **Retrieval declines to be used when priced honestly.** kNN-LM interpolation
   fits λ=1.00 on fold 0 (zero weight on the datastore) when λ is fitted on a
   spatially held-out val; on a *random* val the same code fits λ=0.6 and looks
   worth +0.015. Retrieval alone scores 0.730/0.729/0.721 vs the net's
   0.740/0.742/0.730 — nearly as good, additive with nothing.

### The classical floor (`autoresearch/trials_classic.py`)

Every delta above is measured against the deployed MLP, which answers "better than
what we ship" and never answers "how much of the 0.7414 is the model at all".
Two references on the **same** folds, cls12 relabel, stale-class cleaning,
sqrt class weights and training-fold-only scaler — only the estimator changes:

| model | inner val | F1 | Δ paired | val−test gap |
|---|---:|---:|---:|---:|
| `probe_linear` — logistic regression, C tuned on inner val | 0.7134 | 0.7040 | −0.0373 | 0.009 |
| `rf_tuned` — RF, 6-cell grid on inner val, refit 500 trees | **0.8825** | **0.6945** | −0.0468 | **0.188** |
| **baseline** — deployed 5-seed MLP | 0.7974 | **0.7414** | — | 0.056 |
| *bound_region_oracle* (uses test labels) | — | *0.8144* | *+0.0730* | — |

1. **The deep net is worth +0.0373 over a linear read of AlphaEarth**, paired on
   every fold. The best of 69 scored trials adds +0.0040 on top of that — **9×
   smaller than the nonlinearity itself**, and it halves on a fresh seed. The
   search has been contesting the last ~10% of what the model does.
2. **The nonlinearity is worth most exactly where the search wants to work.**
   Per-class paired deltas vs the deployed net (weak classes in bold):

   | class | 2 rock/sand | 5 grass | 6 scrub | 7 wetland | 12 snow/ice | 8 water |
   |---|---:|---:|---:|---:|---:|---:|
   | `probe_linear` | **−0.087** | **−0.054** | **−0.032** | **−0.047** | −0.034 | −0.008 |
   | `rf_tuned` | **−0.045** | **−0.103** | **−0.013** | **−0.066** | −0.131 | −0.005 |
   | best 6 trials (max) | +0.002 | +0.003 | +0.005 | +0.001 | +0.046 | +0.001 |

   The representation the net builds is already doing the weak-class work — the
   best weak-class gain in the entire search is **+0.005** (scrub), against a
   −0.032 probe deficit on that same class. Three rounds have failed to add to
   *that*, not to some untouched margin. The forest is not a uniformly worse
   probe either: it is **better** on rock/sand and scrub and collapses on
   grassland and snow/ice, the two classes whose boundaries are least
   axis-aligned — and snow/ice is the one class the search does move, so the only
   thing eleven locality mechanisms found is the thing a forest gets most wrong.
3. **A tuned forest fits the training regions far better and transfers worse.**
   Inner val 0.8825 — past the net's 0.7974 and 0.169 past the linear probe —
   then 0.6945 on held-out regions, i.e. **below the linear probe**. Val-to-test
   gap 0.188 vs the net's 0.056. All three folds picked the least-regularised
   grid corner (`max_features=32, min_samples_leaf=1`, ~40M nodes/forest) because
   on a random inner val more fit always looks better. This is the capacity
   finding, the reg sweep and the locality round arriving from a fourth
   direction: in-distribution fit is not the currency, and axis-aligned splits
   are the wrong shape for a surface that must move to an unseen region. It also
   retro-justifies replacing the CatBoost+TabICL stack rather than tuning it.

Cost: probe ~110 s, forest ~95 min (CPU, 8 cores). Both carry pre-flight
self-tests (`python trials_classic.py`) covering class-column alignment and a
poisoned `y_te` that raises if either ever reads the test labels.

### What the arms would cost to deploy (`autoresearch/bench_inference.py`)

The round ranked mechanisms on macro-F1 and stopped there, but nothing ships on
F1 alone — `predict_raster.py` runs the chosen net over ~1.3B px. Measured on the
deployed path (`predict_classmap_gpu`, 5-seed ensemble, 262,144-row chunks, A40):

| arm | params | MAC/row (dense) | *ideal sparse* | M px/s | min / 1.3B px |
|---|---:|---:|---:|---:|---:|
| **baseline** | 51.6k | 51,200 | 51,200 | **8.06** | **2.7** |
| ctrl_capacity | 78.0k | 77,540 | 77,540 | 6.03 | 3.6 |
| mod_depth | 84.9k | 84,224 | *68,332* | 4.54 | 4.8 |
| moe_shared (4 exp) | 78.9k | 78,092 | *64,780* | 2.57 | 8.4 |
| moe_shared (8 exp) | 106.2k | 104,984 | *65,048* | 1.60 | 13.5 |
| moe_shared (16 exp) | 160.9k | 158,768 | *78,896* | 0.91 | 23.8 |

Read against the pipeline's I/O wall (~15 min per 1.3B px at 1.4 M px/s with 6
readers; ~108 min on the CIFS P-drive), because `predict_raster.py` overlaps read
with GPU and the run takes the *longer* of the two:

- **The deployed MLP has 5.8× headroom** — 2.7 min of GPU behind a 15 min read.
  That headroom is why the pipeline is described as I/O-bound above.
- **`moe_shared@8` — the best mean delta in the round (+0.0040) — consumes 87% of
  it.** On this 8-core box it is still hidden by the read, so end-to-end wall time
  barely moves; but it converts a model that was a rounding error into one that is
  co-bound, and the moment reads get faster (more cores, local NVMe, a cached AOI)
  the model becomes the wall. `@16 experts` crosses it outright at 23.8 min.
- **`mod_depth`, the round's only surviving arm, costs 1.8×** (4.8 min) and keeps
  most of the headroom — the cheapest of the three arms that did anything.
- **Top-k buys capacity here, not FLOPs.** `SharedExpertMoE.forward` loops over
  *every* expert and multiplies by a gate weight that is zero for the unselected
  ones, and `MoDNet` computes its extra block for all rows and gates the result.
  The *ideal sparse* column is what a gather/scatter kernel would execute; the gap
  to the dense column is work the current code never skips.
- **Wall time is worse than even the dense MACs**: 5.0× the time for 2.1× the MACs
  at 8 experts. The experts are narrow and each re-reads the whole input block, so
  they run far below the arithmetic intensity of the 256,128 trunk. A sparse
  rewrite would recover part of this, but it is a real cost today, not a
  bookkeeping artefact. *(Superseded — see the next subsection. Most of the gap is
  not arithmetic intensity, it is kernel-launch count, and it comes back without
  any sparsity.)*
- **The MoE degrades to the baseline for free.** `_LocalSwitch.local_off` returns
  the shared expert alone — and `bench_inference.py --selftest` checks it is
  bit-identical to the deployed trunk on every arm, same weights, one flag.
  Measured throughput with it set returns to baseline (8.05–8.77 vs 8.06 M px/s),
  so routed capacity can be dropped at deployment without retraining.
- Without a GPU the ordering is unchanged but nothing is free: 83 min (baseline)
  to 391 min (16 experts) of pure compute.

**Verdict: no arm in this round earns its inference cost.** The +0.0040 that buys
88% of the I/O headroom does not survive a fresh seed; the arm that does survive
(`mod_depth`, +0.0021 seed-averaged) costs 1.8× for a margin at the noise floor.

### That 5.0× was an implementation artefact after all (`autoresearch/moe_fast.py`)

The bullet above priced `moe_shared@8` at 5.0× the baseline forward pass and read
that as low arithmetic intensity. Re-measuring the arm rung by rung says otherwise:
the cost is **kernel-launch count**, and it is recoverable without touching the
mechanism, the weights, or the routing rule.

`SharedExpertMoE.forward` is written for research legibility — a Python loop over
experts, each an `nn.Sequential` of three Linears with hidden dims 64 and 32. Per
ensemble member that is 8 × (3 GEMM + 2 ReLU) plus 8 × (slice, mul, add) ≈ 64
kernels against the shared trunk's 5, and `predict_classmap_gpu` runs five members,
so one chunk dispatches ~350 kernels that each do a few µs of work. The GPU is idle
between them. But every expert sees the same rows, every member sees the same rows,
and all 40 experts are the same shape — so the whole thing is three batched matmuls.
`FusedMoEEnsemble` does exactly that (40 experts + 5 trunks + 5 routers → ~20
kernels) and drops the eval-only routing census.

Ladder from `autoresearch/bench_moe_fast.py`, every rung checked against the
reference loop's class map before it is timed (A40 **shared with another job** — the
absolute rates run ~60% of the table above and vary ~10% between runs, so read the
ratios, not the M px/s; the class-map agreements are bit-stable across runs):

| rung | M px/s | ×arm | class map |
|---|---:|---:|---|
| deployed MLP (reference) | 5.18 | — | — |
| **moe8/2 as written** | **0.89** | 1.00 | exact |
| + drop per-expert host sync / census | 0.86 | 0.96 | exact |
| **+ fuse experts, trunks, routers** | **1.76** | **1.97** | **exact** |
| + TF32 | 2.01 | 2.25 | 0.99958 |
| + fp16 | 2.89 | 3.23 | 0.99906 |
| *(control)* bf16 | 3.52 | 3.94 | 0.99228 |
| + `torch.compile`, fp32 | 2.53 | 2.83 | 0.99931 |
| + fp16 + `torch.compile` max-autotune | 4.08 | 4.56 | 0.99916 |
| experts OFF (fusing's floor) | 7.38 | 8.26 | — |

- **Fusing alone is ~2×, exact.** Not "close enough" — `moe_fast._selftest`
  checks the fused logits against the reference loop, counts route flips at the
  top-k boundary (zero), and requires the ensemble argmax and the emitted class map
  to agree on *every* row. The arm drops from 5.0× the baseline to **2.6–2.9×** —
  carrying that ratio back to the uncontended table above puts it near **6–7 min**
  per 1.3B px, down from 13.5.
- **That puts it above the I/O wall.** 1.76 M px/s is already past the pipeline's
  measured 1.4 M px/s read ceiling *while sharing the GPU with another job*, so the
  arm is no longer the bottleneck of a real raster run: it goes from consuming 87%
  of the I/O headroom to roughly 40% of it. This is why the module stops here and
  deliberately does *not* implement sparse top-k dispatch, which is otherwise a
  genuine further 1.5–2× (see its docstring): it would speed up something that has
  stopped being the wall.
- **The per-expert host sync was a red herring.** `float(we.abs().sum()) == 0.0`
  is a device→host sync per expert per member, and removing it lands inside run
  noise (0.96–1.03×). Launches, not syncs.
- **fp16, not bf16.** bf16 is faster raw (3.52 vs 2.89) but disagrees on the class
  map 8× more often (0.9923 vs 0.9991) — 8 mantissa bits is not enough to preserve
  the ensemble mean-softmax argmax — and once `torch.compile` max-autotune is on,
  fp16 overtakes it anyway (4.08). Both agreement figures are pessimistic: random
  weights with a deliberately sharpened untrained router, so margins are far
  tighter than a trained model's. Neither is needed while I/O is the wall.
- **`chunk` wants to be 65,536, not 262,144.** Dense experts hold a (B, 40, 64)
  activation — 4.0 GB at dnn_core's default — and `--chunk-sweep` shows throughput
  flat from 65,536 up. Four times the memory for nothing.
- **Fusing the ensemble helps the *deployed* model too**: the 5 trunks alone run
  7.38 vs 5.18 M px/s looped, ~1.4× on the plain MLP, which is more than the "+10%"
  recorded in the I/O section below. Free, exact, and independent of the MoE — though
  the deployed MLP already has I/O headroom to spare, so it buys nothing today either.

**This does not change the round's verdict**, which rests on F1: +0.0040 that does
not survive a fresh seed is still not a reason to ship. It changes the *cost* half
of the trade — if a future arm of this shape ever earns its F1, its inference bill
is 2.6× baseline and under the I/O wall, not 5.0× and over it.

### The bound that closes the routing question (`bound_region_oracle`)

Fine-tune one copy per KMeans region, then cheat with the test labels twice:

| | global | best regional (hindsight) | gap | per-row oracle | headroom |
|---|---:|---:|---:|---:|---:|
| fold 0 | 0.7473 | 0.7397 | −0.0076 | 0.8173 | +0.0700 |
| fold 1 | 0.7459 | 0.7406 | −0.0053 | 0.8175 | +0.0716 |
| fold 2 | 0.7309 | 0.7221 | −0.0088 | 0.8084 | +0.0775 |
| **mean** | **0.7414** | **0.7341** | **−0.0072** | 0.8144 | +0.0730 |

**A router that picks one expert per region is capped 0.0072 BELOW baseline before
it makes a single decision — 0 of 12 regional models beat the global model on any
fold.** That is why every hard-routed variant since stage6 has lost; they were not
under-engineered. The per-row oracle's +0.073 is the same shape and size as the
+0.063 ensemble-member oracle that closed the MoE question last time, and is
unreachable for the same reason: it measures disagreement, and disagreement here
doesn't track correctness (cf. `research/diag_ensemble.py`).

### What a local label is worth (`diag_locality.py`)

The deployment has no labels in the target region, so the MoE arms can only
synthesise locality. This diagnostic buys real ones: whole cells inside the
held-out region (a crew visits places, not pixels), scored on evaluation cells
nothing trains on. Δ vs the same fold's global model, mean over 3 folds (±sd):

| cells | ≈ local labels | local model only | global, fine-tuned | pooled into train |
|---:|---:|---:|---:|---:|
| 1 | 4.2k | −0.416 ±.203 | −0.026 ±.023 | **+0.002** ±.001 |
| 4 | 13.5k | −0.257 ±.085 | −0.073 ±.016 | **+0.003** ±.002 |
| 16 | 49.5k | −0.106 ±.021 | −0.052 ±.022 | +0.000 ±.004 |
| 32 | 94.8k | **−0.040** ±.014 | −0.017 ±.008 | +0.000 ±.003 |

**A purely local model converges toward the global one and does not reach it** —
even at 95k local labels, which is most of what the region contains, it is still
0.040 behind a model trained on 440k labels from everywhere else. The gap closes
roughly by half per doubling (−0.416 → −0.257 → −0.106 → −0.040), so parity would
need several times more labels than the region *has*. **Fine-tuning on the local
set is worse than leaving the model alone at every budget** (forgetting; the local
set is small and class-skewed — a 1-cell budget can hold a single class). The one
thing that works is the dullest: **pool the local labels into the global training
set** (+0.002/+0.003 at small budgets, fading to 0.000 once the pool is a rounding
error against 440k existing rows).

For the Vestland/Møre rollout that is the operational answer: **new labels are
worth collecting and worth adding to the pile, and are not worth building regional
models on.** Note the shape of the pooled column — the benefit is largest when the
labels are FEW, because their value is coverage of a region the training set has
never seen, not volume. That is consistent with the learning curves and the
scaling study, and it argues for spreading a survey budget thinly across many
unseen cells rather than densely over a few.

## Feature attention (Stage 9) — also loses, but confirms the key features

With 67 covariates, attention is tempting. Two forms tested:
FT-Transformer feature self-attention (Stage 5, partial run trended <0.7318) and
lightweight feature GATING (`stage9_feat_attn.py`: squeeze-excite + competitive
softmax channel attention in front of the best MLP).

| config | macro F1 | std | vs plain MLP (0.7318) |
|---|---:|---:|---:|
| plain MLP | 0.7318 | .010 | — |
| SE feature-attention | 0.7283 | .010 | −0.0035 |
| softmax feature-attention | 0.7281 | .009 | −0.0037 |

**Both lose** — feature attention genuinely doesn't help (not just an FT-is-heavy
artifact). The 64 AlphaEarth bands are ALREADY a transformer-derived embedding, so
re-attending across features is redundant; the plain MLP's first layer already
learns feature importance implicitly, and explicit gating just adds overfit
variance on the spatial folds.

**But the learned gate weights are informative and corroborate `lidar-features`:**
top-weighted features are **tri, tch, elevation** (the 3 lidar terrain features) +
bands A10/A14/A11/A37/A48; lowest are A55/A56/A43/A26/A04. Attention independently
rediscovered that lidar terrain is the most valuable signal — matching the
CatBoost-importance finding by a completely different method. Acting on it
(reweighting) doesn't help; the MLP already uses those features.

## Class-wise learning curves — problem classes are confusion-bound, not data-starved

`learning_curves.py` / `lc_point.py` (parallel) / `lc_aggregate.py` train the
best recipe on 5%→100% of fold-0 train and plot per-class test F1 vs train size
(`reports/figures/learning_curve.png`). Reading the slope over the last data doubling:

- **Macro F1 saturates by ~70% (310k rows): 0.7326 → 0.7309 at 100%** — a tiny
  dip. More data of the same distribution does **not** help.
- **Grassland (5) and sparse-veg (2) — the two worst classes — are dead flat**
  (5: 0.578→0.585; 2: 0.481→0.495, even dips at 100%). They are
  **confusion-bound**: the ceiling is spectral separability, not label volume.
- open-upland (6), mire (7), forests (3/4), water/bare/built all saturated early.
- **Snow/ice (12)** climbs steeply to ~40% (177k) then peaks/declines — its issue
  is rarity + stale-label noise, fixed by centroid cleaning, not bulk data.

**Implication:** gains must come from better FEATURES that separate the
vegetation classes (lidar already proved this lever) or targeted relabeling of
noisy classes — not more data, and not a bigger/fancier network. This is also
why every alternative architecture (GLU/SNN/FT/MoE) failed: you cannot model past
genuine feature-space overlap.

## Targeted class-12 relabel — NEW BEST (Stage 8)

cls12 (snow/ice) is the ONE noise-limited class, and its noise is static
mislabelling: 29/146 locations are suspect in *every* year 2017-2025 (no temporal
trend) — permanently mislabelled rock/debris polygons, not deglaciation. Blanket
relabel banks cls12 but wrecks grassland/mire (−0.0065 macro). So
`stage8_cls12_relabel.py` applies leak-free perfold corrected labels to cls12-
touching rows ONLY:

| config | macro F1 | cls12 | grassland/5 | mire/7 |
|---|---:|---:|---:|---:|
| best (centroid clean) | 0.7318 | 0.702 | 0.599 | 0.691 |
| cls12_fix (recode bad ice) | 0.7332 | 0.724 | 0.598 | 0.690 |
| **to12_fix (also recover→12)** | **0.7341** | **0.726** | 0.599 | 0.691 |

**to12_fix is the new best: +0.0023 macro, +0.024 cls12, grassland/mire untouched**
— exactly the surgical win the diagnostics predicted. Caveat: cls12 is rare and
fold-2 (207 test rows) is volatile (cls12 there *dropped* to 0.61 while folds 0/1
rose to 0.77-0.79), so the cls12 gain is real-but-noisy; macro std unchanged
(0.009). Headroom beyond this needs external ground truth (Sentinel-2 NDSI
persistence / NVE glacier inventory), parked.

## Per-class label-noise diagnosis (leak-free)

Four tests separate *label noise* (fixable by relabeling) from *feature overlap*
(irreducible) per class. All leak-free, reusing the per-fold cleanlab artifact
(`clean_labels_perfold.npz`, test labels never touched). Scripts:
`noise_diagnostics.py` (A/B/C), `noise_clean_gain.py` (D).

- **A. Transition matrix** — est. label-noise rate per class (1 − P(true=c|obs=c)).
- **B. Confident-disagreement** — % of a class's rows the DNN confidently contradicts.
- **C. Noise-corrected ceiling** — how much low F1 is just noisy scoring.
- **D. Clean-vs-noisy F1 gain** (decisive) — retrain on leak-free corrected labels;
  per-class F1 Δ. Rise ⇒ noise-limited; flat/drop ⇒ confusion-bound.

| class | F1 | noise% (A) | confDis% (B) | relabel Δ (D) | verdict |
|---|---:|---:|---:|---:|---|
| sparse-veg/2 | 0.55 | 19% | 11% | −0.00 | confusion-bound |
| **grassland/5** | 0.60 | 20% | 9% | **−0.022** | **confusion-bound** (relabel HURTS) |
| open-upland/6 | 0.66 | 19% | 10% | +0.00 | confusion-bound |
| mire/7 | 0.69 | 19% | 4% | **−0.032** | confusion-bound (relabel HURTS) |
| **snow-ice/12** | 0.70 | 13% | 15% | **+0.037** | **NOISE-LIMITED** → cls11 |
| forest/3,4 | 0.74-0.79 | 14% | 4% | −0.01 to −0.02 | confusion-bound |
| bare/10, water/8 | 0.85-0.95 | 2-7% | 1-3% | ~0 | clean & easy |

**Verdict: label noise is a real, fixable problem for ONE class — snow/ice (12)**,
whose labels are systematically stale (glacier/snow → bare-rock/infrastructure,
both noise and model arrows point to class 11). Centroid cleaning already
exploits this. For the worst classes (grassland 5, sparse-veg 2, mire 7), noise
is NOT the issue — cleanlab relabeling makes them *worse* (it recodes genuine
grassland as forest because they overlap), confirming the flat learning curves:
they are confusion-bound. **Bulk relabeling hurts the DNN macro-F1 (−0.0065)** —
note this differs from TabICL where perfold_fix gave +0.007, because the DNN
trains on the full 664k rows with class weighting + ensembling and is already
more noise-robust than the 32k-support TabICL.

## Grunnkart v2 (FKB Bygning 2018) — a MAP fix, not a MODEL fix

`GIS/NIBIO/Version_2/rasterized_10m/grunnkart_nyvest_10m_v2.tif` swaps FKB
Grønnstruktur's `FKB_bygg` for FKB Bygning 2018. Verified pixelwise against v1
(`relabel_v2.py`, `extract_v2_changes.py`): **118,968 pixels change, 0.0066% of
the grid, and every one of them changes *to* class 10 (built)** — +1.36% built,
−1.46% "other" (13), all other classes ≤0.07%.

| from | to | pixels | km² |
|---|---|---:|---:|
| 13 other | 10 built | 102,677 | 10.27 |
| 4 forest | 10 built | 4,937 | 0.49 |
| 6 scrub | 10 built | 3,792 | 0.38 |
| 3 crop | 10 built | 2,492 | 0.25 |
| *(8 more, all → 10)* | | 5,070 | 0.51 |

**The training set barely notices, by construction.** 86% of the edit is class
13, and FSCS sampling excludes 13 entirely (`CLASSES = 1..12`). Sampling both
rasters at the 74,639 training points: local v1 agrees with the parquet labels
**100.00%** (method check), and only **3 points — 27 of 663,740 point-year rows
(0.004%) — change label**.

### The experiments (`exp_v2_labels.py`, 3 paired seeds, reference 3-fold CV)

| arm | macro-F1 (3 seeds) | paired Δ vs v1 |
|---|---|---:|
| v1 labels | 0.7352 / 0.7339 / 0.7328 → **0.7339** | — |
| v2 labels | 0.7347 / 0.7330 / 0.7336 → 0.7338 | **−0.0002** (sd 0.0009) |
| v1 + 5k newly-built rows | 0.7321 / 0.7327 / 0.7317 → 0.7322 | **−0.0017** (sd 0.0011) |

Swapping the labels is a **clean null** (−0.0002, well inside the sd-0.0013
noise floor) — as the 3-point churn guarantees. Adding the newly-built pixels as
extra class-10 training rows is **negative on all three seeds**, and the damage
is concentrated in the class it was meant to help: **built/10 F1 −0.0095**
(every other class moves ≤0.002 except the tiny, volatile cls12).

### Why: the audit against the corrections (`audit_v2.py`, 76k pixels)

Retraining can't move, so ask the inverse question — on the pixels v2 newly
calls built, what did the deployed model already predict? Scored with real
lidar and 100% AlphaEarth coverage:

| stratum | n | predicted built | mean p(built) |
|---|---:|---:|---:|
| changed 13→10 | 20,000 | 39.1% | 0.304 |
| changed (1..12)→10 | 16,291 | 18.7% | 0.154 |
| ctrl: unchanged 13 | 20,000 | **43.8%** | 0.340 |
| ctrl: unchanged 10 | 20,000 | 82.8% | 0.692 |

The control decides it: **unchanged class-13 scores *higher* built (43.8%) than
the pixels v2 corrected (39.1%)**. The correction did not target pixels the
model finds unusually built-like — class 13 ("settlements & artificial areas")
is already ~44% built-looking to the model throughout.

Conditioning on local built density (5×5 neighbourhood) gives the mechanism:

| 5×5 built density | changed 13→10 | ctrl unchanged built |
|---|---:|---:|
| isolated (<3/25) | 22.1% | 40.2% |
| sparse (3–6/25) | 59.6% | **59.9%** |
| clustered (7–12/25) | 85.4% | **86.3%** |
| dense (>12/25) | 98.3% | **96.0%** |

**Match on density and the newly-labelled built pixels are recognised at the
same rate as established built.** The v2 additions are simply 67% isolated
(vs 8% for established built) — scattered rural buildings, one or two 10 m
pixels each, spectrally dominated by the grass or forest around them. That is a
resolution limit, not a model error, which is exactly why training on them
*hurts*: it teaches the net to call grass-dominated mixed pixels "built".

**Verdict: adopt v2 for the map, keep the model as-is.** v2 is a genuine
cartographic improvement (10.3 km² of correctly attributed rural building) that
the 10 m feature set cannot exploit as training signal. Do NOT augment training
with the changed pixels. Recovering isolated rural buildings needs a
higher-resolution or building-specific input (FKB footprints as a feature, or
sub-10 m imagery), not more labels.

**Gotcha banked:** `aef_2024.vrt` is float32 and **already unit-normalised**
(measured L2/px = 1.0001, identical to the training parquet). The `--scale 1000`
convention applies to the int-scaled source.coop / GEE-export tiles, *not* here
— applying it drives every feature to ~0 and silently produces garbage
predictions. `extract_v2_changes.check_embedding_scale()` now raises if the
median L2 strays from 1.0; call it after any new embedding source is wired in.

## Files

- `data_utils.py` — shared loader, spatial-fold splitter, cls12 centroid
  cleaning, macro-F1. Mirrors the reference pipeline exactly.
- `stage1_mlp.py` — plain MLP baseline.
- `stage2_resmlp.py` — residual+BN MLP (capacity probe; no gain).
- `stage3_robust_mlp.py` — **the model** (winning config). Knobs are env vars.
- `ablation_one.py` / `run_ablation.sh` — Stage-3 lever ablation.
- `conformal_utils.py` + `stage4_pseudo.py` — CC-APS pseudo-labelling (no-op).
- `stage5_arch.py` — architecture sweep (glu/residual_glu/snn/ft via ARCH env).
- `stage6_moe.py` — soft Mixture-of-Experts (added by subagent).
- `learning_curves.py` (serial) / `lc_point.py` + `lc_aggregate.py` (parallel,
  one process per (frac,seed) point) — class-wise learning curves +
  `reports/figures/learning_curve.png`.
- `autoresearch/` — the mechanism search (rounds 1–3). `trials_classic.py` holds
  the classical floor (linear probe, tuned RF) and `bench_inference.py` prices any
  arm on the deployed inference path; both carry pre-flight self-tests, see
  `autoresearch/README_LOCALITY.md`.
- `relabel_v2.py` / `extract_v2_changes.py` / `audit_v2.py` / `exp_v2_labels.py`
  — the grunnkart v2 round: sample both rasters at the training points, pull the
  changed pixels with AlphaEarth + lidar, audit the deployed model against the
  corrections (with density strata and controls), and run the paired
  relabel/augment CV. See the v2 section above.
- `dnn_paths.py` — shared path constants. Generated experiment outputs are kept
  under `reports/results/`, `reports/logs/`, `reports/figures/`, and
  `reports/html/` so the top-level folder stays source-focused.

## Run

```bash
PY=~/myprojects/recover/.venv/bin/python
# canonical winning config (all defaults)
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/stage3_robust_mlp.py

# grunnkart v2 round (in order; ~10 min each for the raster passes)
$PY DNN/relabel_v2.py                              # label churn at train points
$PY DNN/extract_v2_changes.py --max_points 30000   # changed px + AE + lidar
$PY DNN/audit_v2.py --n_ctrl 20000                 # model vs the corrections
$PY DNN/exp_v2_labels.py --exp both --seeds 0,1,2  # paired relabel/augment CV
```

GPU memory: trivial (<1 GB) — the whole frame is 664k×67 float32 and the net is
tiny. No need for the chunking/`TOTAL_SUP` ceilings that bound TabICL.

## Refactored library — fast iteration + deployable inference

The `stageN` scripts above are the *experiment record*; each grew its own copy of
the scaler → val-split → ensemble → predict loop. `dnn_core.py` consolidates that
into one fast, reusable core so a new idea is ~30 lines and a chosen recipe can be
baked into a single portable artifact for production inference on AlphaEarth
rasters.

### `dnn_core.py` — the shared core
Same winning maths (class-weighted CE, label smoothing, early-stopped 2-layer MLP,
5-seed probability ensemble, StandardScaler), verified to reproduce the reference
fold-0 F1 (0.737). Speed/ergonomics over the old per-stage loops:

- **frame cache** (`load_cached`): memoises the duckdb parquet load to `.cache/`,
  self-invalidating on parquet mtime — repeat runs go **2.1 s → 0.2 s**.
- **whole dataset on-GPU**, no per-epoch H2D copies.
- **on-GPU per-epoch val macro-F1** (`gpu_macro_f1`, bit-matches sklearn) — kills
  the `.cpu()`+sklearn call every epoch.
- **`Config`** dataclass — every hyperparameter, all env-overridable (same env
  names as stage3), so sweeps stay one-liners.
- **`Ensemble`** — a fitted object you can `predict_proba`/`predict`/`save`/`load`;
  save is bit-exact on reload. This is the deployable model.
- **`cv_evaluate(data, cfg, relabel_fn=...)`** — the reference 3-fold spatial CV in
  one call; `relabel_fn` injects per-fold train-only label surgery (e.g. stage8).

**Note on AMP / `torch.compile`:** both are wired (`DNN_AMP=1`, `DNN_COMPILE=1`)
but **default OFF** — measured *slower* on this tiny 256→128 MLP (fp16 cast +
GradScaler overhead beats the tensor-core gain; single-fold 20.6 s → 34.3 s with
AMP, and slightly lower F1). The workload is Python-minibatch-loop-bound on a tiny
net, so the real levers are the cache + on-GPU eval, not lower precision. Keep the
switches for any larger architecture probed later.

New experiment sketch:
```python
import dnn_core as C
data = C.load_cached("lidar")
res  = C.cv_evaluate(data, C.Config(hidden=(512, 256)))   # your idea here
```

### `train_final.py` — bake the recipe into one artifact
Trains the winning recipe (5-seed ensemble + `to12_fix` cls12 relabel, the only
label fix that helped) on the **entire** stable frame — no held-out fold, so it
uses the full-data cleanlab artifact `clean_labels_full.npz` (no test to leak
into). Writes `models/dnn_final.pt` (weights + scaler stats + class decode map +
feature order + lidar medians) and a `.meta.json`.
```bash
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/train_final.py            # RELABEL=to12_fix EXTRA=lidar by default
```

### `fit_calibration.py` — uncertainty quantification (calibrated proba + conformal sets)
Companion to `train_final.py`. `train_final.py`'s own training probabilities are
optimistic (trained on everything, no held-out fold) so they can't be used to
calibrate uncertainty. This script instead runs the reference leak-free 3-fold
spatial CV to produce out-of-fold (OOF) probabilities over the whole stable
dataset, then fits two things:

- **LAC + Mondrian (class-conditional) conformal prediction sets**, alpha=0.1
  (90% target coverage) — the method Investigation 3 (`conformal_compare.py`)
  found best: tightest sets among valid methods, no empty sets, per-class
  coverage spread 0.29→0.035 vs plain split (see the Conformal UQ section
  above).
- **Point calibration**, head-to-head between temperature scaling (Investigation
  3's method) and Venn-Abers (one-vs-rest binary calibrators — the `venn-abers`
  package is binary-only, so multiclass here means one calibrator per class,
  fit on `[1-p_c, p_c]` vs `y==c`, renormalized at inference). Both are scored
  by ECE on a held-out half of the OOF set; the lower-ECE method is saved and
  used for the `pcal_*` output columns/bands. Both numbers are recorded in
  `dnn_final_calib.meta.json` regardless of which wins.

Writes `models/dnn_final_calib.npz` (+ `.meta.json`), consumed by `predict.py`
and `predict_raster.py`'s `dnn_core.Calibration`.
```bash
systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 \
  $PY DNN/fit_calibration.py        # ALPHA=0.1 EXTRA=lidar RELABEL=to12_fix by default
```

**Result (OOF macro-F1 0.7321, close to the 0.7341 CV reference): Venn-Abers
wins decisively.**

| method | ECE before | ECE after | fit time | apply time (331,870 rows) |
|---|---:|---:|---:|---:|
| temperature scaling (T=0.78) | 0.0694 | 0.0394 | 18s | 12.8s |
| **Venn-Abers OvR** | 0.0694 | **0.0042** | 485s | **1.6s** |

Venn-Abers is ~9x better calibrated (ECE 0.0042 vs 0.0394) AND ~8x faster to
*apply* (the isotonic-regression fit is a one-time ~8 min cost at calibration
time; inference is a vectorized `searchsorted`, cheaper than temperature
scaling's softmax). It won on both axes, so `dnn_final_calib.npz` uses
Venn-Abers for the `pcal_*` columns/bands. LAC+Mondrian conformal sets: 90.0%
coverage (target), avg set size 1.55, singleton rate 57%, per-class coverage
tight to 0.9000–0.9020 — matches the Investigation-3 reference behavior.

### `predict_raster.py` — inference on AlphaEarth GeoTIFF stacks
Block-windowed (bounded memory) inference on 64-band AE tiles (band order
A00..A63, matching the GEE/source.coop export contract). Writes a single-band
classified GeoTIFF of **raw class codes** (e.g. 12 = snow/ice) + optional
`--uq-out` multiband GeoTIFF: per-class calibrated probability + prediction-set
size + per-class 0/1 conformal-set-inclusion, `2C+1` float32 bands with
self-describing band names (`pcal_c{v}`, `set_size`, `inset_c{v}`). Nodata
pixels → `--nodata-class` (default 0). If the model uses lidar, pass
`--lidar-raster` (bands elevation,tri,tch on the same grid) or omit it to
median-fill from the saved training medians.
```bash
$PY DNN/predict_raster.py --in ae_tile.tif --out classified.tif \
    [--uq-out uq.tif] [--lidar-raster lidar.tif] [--scale 10000]
```
`--scale` divides the AE bands if a source packs them as scaled ints; omit for the
float embedding used in training.

### `predict.py` — inference on tabular AE points (parquet)
Same artifacts, for point/parquet inputs shaped like the training data (columns
A00..A63 [+ lidar]). Missing lidar columns are median-filled. Always writes
`pred_class` + the UQ columns (`pcal_{c}`, `set_size`, `inset_{c}`).
```bash
$PY DNN/predict.py --in points.parquet --out preds.parquet
```

## Inference performance & wall-to-wall county scaling

Target use case: classify 3 Norwegian counties wall-to-wall at 10 m ≈ **1.3
billion pixels**. Measured facts (A40, shared) that shape the design:

| stage | throughput | 1.3B px |
|---|---:|---:|
| **model compute only** (5-seed ensemble, on-GPU) | ~6 M px/s | **~4 min** |
| fused batched-bmm ensemble vs python loop | +10% | (compute isn't the wall) |
| `torch.compile` on the forward | fragile / OOM-prone, ~10% | not worth it |
| **reading + decompressing** a 64-band deflate tile (1 handle) | **~0.7 M px/s** | **~30 min** |

**The model is not the bottleneck — raster I/O (decompression) is.** A 64-band
float32 tile is huge (a 3000×3000 tile = 695 MB compressed), and decompression, not
the MLP, dominates. So `predict_raster.py` is a **producer/consumer pipeline** that
overlaps read → GPU → write and, crucially, gives each reader thread its **own GDAL
handle** so decompression parallelizes across cores (a shared handle serializes
reads and kills the gain):

| readers | end-to-end throughput |
|---:|---:|
| 1 | 0.5 M px/s |
| 3 | 1.2 M px/s |
| 6 | 1.4 M px/s (8-core box; read-ceiling) |

Results are **bit-identical across reader counts** (deterministic). On the 8-core
VDI, `--readers 6` gives ~2.8×; the ceiling is the machine's decompression
bandwidth, not the GPU (which idles most of the run).

### Can the read wall move? (`bench_io_formats.py`)

With the model side fused the read is the only wall left, so it is worth asking
what sets it. It is **not** the bytes on disk — the prepped tiles compress to
~0.67 bytes/value. It is that we chose to store 8-bit data in 32 bits.

`prep_aef_tiles.py` reads source.coop AEF tiles that are **int8 + ZSTD**, warps
them with *nearest* resampling, and writes **float32 + DEFLATE**. Its
dequantisation is a 256-entry LUT (`_DEQUANT_LUT`) and nearest resampling cannot
invent a 257th value, so the prepped raster provably carries 8 bits per value
(measured on a real window: 178 distinct values across 64 bands) and spends 32
storing it. DEFLATE hides that on disk, but every read still has to materialise
4 bytes/value with zlib — CPU the pipeline cannot overlap away.

Every variant below is verified to decode back to the *original* float32 values
before it is timed. 1536² window, 64 bands, 6 readers, 8-core VDI:

| encoding | MB | B/value | warm | cold | vs today |
|---|---:|---:|---:|---:|---:|
| **float32 DEFLATE 512** *(what prep writes)* | 100.5 | 0.67 | 13.7 | 11.9 | 1.00× |
| float32 DEFLATE +predictor=3 | 306.8 | 2.03 | 6.6 | 5.2 | 0.48× |
| float32 ZSTD 512 | 113.0 | 0.75 | 12.4 | 10.0 | 0.90× |
| int8 ZSTD 512, CPU dequant | 80.0 | 0.53 | 8.2 | 7.6 | 0.60× |
| **int8 ZSTD 512, GPU dequant** | 80.0 | 0.53 | **54.9** | **29.8** | **4.01×** |

- **int8 + ZSTD with the LUT moved to the GPU is ~4× warm, ~2.5× cold**, and the
  files are 20% smaller. The LUT expansion does not vanish, it moves: measured at
  **299 M px/s on-GPU**, ~6× faster than the fastest read, i.e. free. The pipeline
  already ships every block to the GPU, so it is the natural place for it.
- **The dtype is the lever, not the codec.** float32 ZSTD alone is 0.90× — no
  help. int8 DEFLATE is already 2.16×. Storing 8 bits in 8 bits is the whole win.
- **The dequant must move to the GPU or the change backfires**: doing the LUT in
  numpy in the reader thread gives 0.60×, *slower than today*.
- **`predictor=3` is actively harmful** — 0.48× and a 3× bigger file. Float
  predictors assume smooth floats; these are 178 quantisation levels, where
  byte-level entropy coding already wins.
- **Match the internal block size to the read window.** int8 ZSTD with 1024-px
  blocks read in 512-px windows falls from 4.01× to 1.77×, because each read
  inflates four blocks to use one.
- **Reader threads and `GDAL_NUM_THREADS` are substitutes, not multipliers**
  (`--thread-sweep`). GTiff defaults to single-threaded decompression, worth ~4×
  on its own — but the pipeline already runs 6 readers, so both knobs are ways of
  spending the same 8 cores and they saturate at the same ceiling. That ceiling
  is set by the encoding: **14.3 M px/s for float32 DEFLATE, ~55 for int8 ZSTD**,
  and no thread setting moves either.

The change is small: `prep_aef_tiles.py` already reads warped int8 and *then*
dequantises (`dst.write(dequantize_lut(raw))`), so it would write `raw` as int8
with `compress="zstd"` and nodata −128; `predict_raster.py` gains a `torch.take`
on the 256-entry LUT after the upload it already does. The cost is reprocessing
the prepped tiles once.

**Caveat on the absolute numbers**: these are one stage — decompress a window of
one local tile, no warp, no GPU, no write — on a window that is still ~25%
nodata (this bench tile is a single warped UTM tile; production reads a VRT
mosaic where interior windows are dense). The ratios are the result; the M px/s
are an upper bound on the read stage alone. Two confounds had to be removed to
get even those: GDAL's block cache was serving the timed reads as memcpy
(`GDAL_CACHEMAX=32`), and a `--size`/`--block` pair yielding a single window made
the reader-thread column a copy of the single-thread column (now an error).

### What is actually deployed now (`verify_deploy.py`)

The optimisations above are wired into the pipeline and verified end to end.

| | artifact / setting |
|---|---|
| model | `models/dnn_final_moe8.pt` — 5-seed `moe_shared`, 8 experts, top-2, val_f1 0.8019 |
| execution | `moe_fast.FusedMoEEnsemble`, built lazily by `Ensemble._fused()` |
| GPU chunk | `Ensemble.default_chunk` = 65,536 for a MoE (262,144 for an MLP) |
| raster reads | `GDAL_NUM_THREADS=ALL_CPUS`, set before rasterio imports GDAL |
| tiles | `prep_aef_tiles.py --int8` → int8 + ZSTD, 512 blocks; `predict_raster.py` detects the dtype and runs the LUT on the GPU |

`dnn_core.Config.arch` picks the architecture and `Ensemble.save/load` carry it, so
`models/dnn_final.pt` still loads as an MLP (checked with `ARCH=moe_shared` in the
environment — an old checkpoint must not be reinterpreted by ambient state), and
reverting is `--model models/dnn_final.pt`.

**Full tile, 8924² = 79.6M px, cold cache, `--readers 6`** (pipeline time; the
second number includes process start and model load):

| | pipeline | total | vs old |
|---|---:|---:|---:|
| **(a) old: MLP + float32 tile** | 20.0 s | 24.5 s | 1.00× |
| **(b) new: MoE + float32 tile** | 21.6 s | 26.5 s | 0.93× |
| **(c) new: MoE + int8 tile** | **11.5 s** | **16.3 s** | **1.74×** |

- **The MoE now costs ~8% end-to-end, not 5×.** That is the fusion doing its job:
  before it, this arm consumed 87% of the I/O headroom.
- **int8 tiles nearly halve the wall time** (b → c is 1.88×), and pay for the
  model upgrade several times over. This is the one change that needs
  reprocessing; everything else is live already.
- All three runs classify the same 8,189,525 valid pixels.

`verify_deploy.py` checks the claims that the component benchmarks could not,
9/9 passing:

- **fused == loop on the TRAINED weights**, 400k real embedding rows: argmax
  agreement 1.000000, max |Δp| 2.4e-07. `moe_fast._selftest` only proves this for
  random weights, and random weights have flat routers — a trained router is
  confident and puts far more rows near the top-k boundary, which is exactly
  where a reassociated GEMM could flip a route. It does not.
- **int8 tile == float32 tile, bit-identical class maps**: 2,359,296/2,359,296 on
  the subset and 79,637,776/79,637,776 on the full tile, with identical validity
  masks. The masks survive because every test in the reader (`== nodata`, `== 0`)
  is order-preserving under a strictly monotone LUT that maps 0 → 0.0.
- **`prep_aef_tiles.py --int8` is lossless**: a fresh int8 prep and a fresh
  float32 prep of the same raw tile are *exactly* equal after the LUT (max|Δ| 0).
- The MoE changes **6.5%** of valid pixels versus the MLP — the map really did
  move, which a run that silently fell back to the baseline would not show.

Two things found while verifying, neither introduced by these changes:

- `aef_bench_prepped/…_32633.tif` (Jul 9) is **not reproducible** from the current
  `prep_aef_tiles.py`: a fresh prep of the same raw tile differs in **2 pixels of
  2.36M** (114 band-values, max |Δ| 0.083) — a nearest-resampling tie-break at a
  warp boundary. Harmless at this magnitude, but it means that tile is stale.
- `torch.compile` + fp16 remains the fastest model config measured (4.6× the
  unfused arm) and is **not** enabled: it is not exact, and with I/O now the wall
  it would buy nothing. `bench_moe_fast.py` keeps the number if that changes.

### Recommendations for the county-scale run
- **Tile the counties** (e.g. a manifest of ~5–10k px tiles) and run one process
  per tile batch across machines/cores — the pipeline scales with **read cores**,
  so more cores/nodes beats a faster GPU. A single 8-core box does 1.3B px in
  ~15–25 min of wall time; more read cores shorten it proportionally until the GPU
  (~4 min) becomes the floor.
- **Source-agnostic**: `--in` accepts any rasterio path — local GeoTIFF, VRT, or a
  COG URL (`/vsicurl/…`, GCS, S3). For source.coop AE COGs, block reads become
  HTTP range requests; more `--readers` then overlaps network latency too.
- **Skip `--uq-out`** unless you need it — the class map is one int16 band; the
  UQ stack is `2C+1` float32 bands (21 for the 10-class model, 20×+ the write
  volume + memory).
- **AMP / torch.compile stay off** — measured slower / OOM-prone on this tiny net.
- If read bandwidth ever stops being the wall, distilling the 5-seed ensemble into
  one net drops compute 5× (but compute is already ~4 min, so it's rarely worth
  the F1 risk).
- **Any architecture change has to be priced here, not just on F1.** The headroom
  above is what a bigger net spends: the deployed MLP uses 2.7 of the ~15 min the
  read takes, an 8-expert shared MoE uses 13.6, and a 16-expert one exceeds the
  read outright. `autoresearch/bench_inference.py` measures any candidate on this
  exact path — see "What the arms would cost to deploy" in the locality round.

```bash
# fast wall-to-wall tile (local or COG URL), 6 read threads:
$PY DNN/predict_raster.py --in ae_tile.tif --out classified.tif --readers 6 --block 2048
```

## Inference data sourcing — the P-drive embeddings don't cover the AOI

`embeddings.vrt` on the P-drive is framed to the full 3-county AOI (15002×18002
px) but only 154 of the tiles that would fill it actually exist — most of the
mosaic is empty. `DNN/fetch_aef_sourcecoop.py` investigates 3 ways to get
AlphaEarth embeddings for the missing area (tested on a 2024 subset):

| option | cost/auth | throughput (measured, apples-to-apples on a 20×20km/2000×2000px/64-band 2024 window) | full AOI (270M px) | verdict |
|---|---|---:|---:|---|
| **source.coop, 1 handle, streaming reads** | free, no auth | ~184k px/s (1 open, 4 sequential sub-window reads) | **~24 min** | **winner, and scales further with parallel readers (untested)** |
| **source.coop, 1 handle, single big read** | free, no auth | 124k–152k px/s | ~30–36 min | good, simpler code |
| **geedim `img.gd.toGeoTIFF()`** | free (existing `ee-gsingh` project) | 98.3k px/s (`max_requests=100`); 57.6k px/s at default `max_requests=32` | 46 min / 78 min | easiest to wire up, no UTM/dequant handling |
| source.coop, reopen handle per window | free, no auth | 85k px/s (4× reopen cost dominates) | ~53 min | **anti-pattern — don't reopen per block** |
| GEE `ee.data.computePixels` (raw, hand-tiled) | free | ~12k px/s (48 MB/request cap forces ~290×290 px tiles) | ~6 hr single-threaded | superseded by geedim |
| GCS `alphaearth_foundations` | requester-pays, needs billing + creds | not measured — anonymous access denied | — | blocked on credentials |

**Correction to an earlier pass in this investigation:** an initial source.coop
estimate of "~2 min for the full AOI" was extrapolated from small (512×512px)
windows and was wrong — retested at the same 2000×2000px scale geedim was
benchmarked at, single-connection source.coop is **~30 min**, not ~2 min.
Still faster than geedim but only ~1.3–1.9×, not ~20×. Small-window
extrapolation was misleading; always benchmark at the scale you'll actually
run at.

**Streaming (1 persistent handle, many reads) vs. reopening per request
matters a lot, and directly favors source.coop's architecture:** opening a
fresh `rasterio.open()` + `WarpedVRT` handle costs real fixed time (TIFF
header/IFD fetch + warp-transformer setup). Reopening once per window to
cover 4 sub-tiles of the test area took 46.7s total; keeping ONE handle open
and reading all 4 sub-windows through it took 21.6s (2.2× faster). This is
exactly the pattern `predict_raster.py`'s reader threads already use (one
GDAL handle per thread, reused across all its assigned blocks) — so wiring
source.coop in via that existing architecture, rather than a naive
open-per-tile downloader, should land close to the streaming number above.
**Not yet tested: multiple parallel reader threads each streaming from their
own handle** (`predict_raster.py` already supports `--readers N` for local/
COG sources) — this is the highest-value next experiment, since the existing
6-reader local-tile benchmark showed ~2.8× over 1 reader before hitting the
machine's decompression ceiling, and source.coop's bottleneck (network RTT +
zstd decompression) may parallelize similarly.

**geedim** (`ee.Image.gd` accessor, v2.0 API) wraps the `computePixels`
tiling problem: `img.gd.prepareForExport(crs=, scale=, region=)` then
`prepped.gd.toGeoTIFF(path, max_requests=N)` auto-splits the region into
GEE-sized tiles and downloads them with a thread pool, writing directly into
one GeoTIFF. Tested against the same 2024 AlphaEarth mosaic
(`.reduce(ee.Reducer.first())`, matching `sample_feature_space_stable_allyears
.build_alphaearth_year`): `max_requests=32` (default) gave 57.6k px/s;
`max_requests=100` gave 98.3k px/s (1.7×) with no quota errors — GEE's
per-project concurrent-request quota didn't bite at that level on this small
test, but a sustained ~30-46 min run at that concurrency was NOT tested and
could throttle differently. Output is **already reprojected to 32633 and
analysis-ready float64** (unit-norm verified, L2=1.0000516) — no
dequantization or WarpedVRT reprojection step needed, unlike source.coop. Also
uses the collection's per-pixel `.reduce(first())` mosaic directly, so tile
seams are handled by GEE server-side (source.coop tiles are the *raw*,
non-mosaicked per-UTM-zone EE images — cross-zone seam behavior vs. the
collection's mosaic was not verified either way in this pass).

**Key findings:**
- source.coop (`data.source.coop/tge-labs/aef`) mirrors the exact GCS AEF
  dataset over plain HTTPS, no auth, `accept-ranges: bytes` (vsicurl-friendly).
  An index (`aef_index.parquet`, 302k rows, ~78 MB) maps year + WGS84 bbox +
  UTM zone → tile URL — filter it instead of scanning directories.
- **GDAL's default vsicurl config is ~5× too slow on this dataset**: the COGs
  are band-interleaved (64 separate per-band blocks), and without HTTP/2
  multiplexing GDAL fetches them serially (41 s for a 1024×1024×64 block).
  Setting `GDAL_HTTP_MULTIPLEX=YES` + `GDAL_HTTP_VERSION=2` drops that to 8 s
  (~5×) and pushes small-window (512×512) throughput to ~140 MB/s. This is
  the single biggest lever and is easy to miss.
- Data is **int8-quantized** (dequantize: `((v/127.5)**2)*sign(v)`, −128 =
  nodata — see `dequantize()`) and the raw `.tiff` is **bottom-up**
  (row 0 = south). The dataset ships a per-tile `.vrt` that corrects this,
  but it hardcodes `/vsis3/...` (needs `AWS_NO_SIGN_REQUEST=YES`, works fine)
  and its `VRTWarpedDataset` warp is itself slow (41 s for a 512×512×64
  window vs 2–4 s reading the raw `.tiff` directly) — **the warp machinery is
  the cost, not the transport.** Fastest correct path: read the raw `.tiff`
  via `/vsicurl`, flip the row axis and dequantize in numpy.
- The AOI straddles **UTM zones 31N and 32N** in the source data (only 20
  tiles total cover the 3 counties for one year), neither matching the
  project's working CRS (32633). Reprojecting via a static warped-VRT file
  is slow for the same reason above; `rasterio.vrt.WarpedVRT` used
  in-process at read time is fast (0.09 s/window once GDAL's block cache is
  warm) and correctly handles the flip + reprojection in one pass — this is
  what `open_tile_window()` does. Verified end-to-end (source.coop tile →
  `WarpedVRT` reproject → dequantize → `dnn_core.Ensemble` inference) on a
  live 1024×1024 window: dequantized L2 norm 1.000 ± 0.01 (correct unit-norm
  embeddings), predictions ran in <1 s on GPU.
- GEE's synchronous `ee.data.computePixels` is capped at exactly 50,331,648
  bytes/request (48 MiB) and serves **float64**, not int8, so a request maxes
  out around 290×290 px for 64 bands. **geedim solves this properly** — see
  above — by tiling + thread-pooling the same `computePixels` calls
  internally, reaching 98.3k px/s at `max_requests=100` vs ~12k px/s hand-
  rolled. `Export.image.toCloudStorage`/`toDrive` (async batch, server-side
  tiled, no 48 MB cap) was not benchmarked — it avoids the per-request cap
  but queues as a background job (minutes–hours latency), unsuitable for
  interactive iteration; worth using for a one-time bulk backfill export
  instead of point-in-time streaming, if source.coop's 20-tile-per-year
  footprint is ever unavailable or wrong.
- GCS direct access (`/vsigs/...`) failed with `AWS`-equivalent anonymous
  access — the bucket is requester-pays and needs a real billing-linked GCP
  project (`GS_NO_SIGN_REQUEST` does not bypass this, unlike source.coop's
  public S3 mirror). Not pursued further since source.coop serves identical
  data for free.

**Recommendation:** for a **one-off test pull**, geedim is the path of least
resistance — `pip install geedim`, reuses the existing `ee-gsingh` auth, one
function call gives an analysis-ready reprojected GeoTIFF, ~46 min for the
full AOI at `max_requests=100`. For **`predict_raster.py`-style streaming
inference** (read-infer-write per block, no full download), source.coop with
persistent per-thread handles is faster (~24–36 min full-AOI equivalent,
single-threaded, untested with multiple parallel readers) and
infrastructure-free, at the cost of handling the UTM-zone reprojection
(`WarpedVRT`) and int8 dequantization yourself — wire `open_tile_window`-style
per-tile reads into `predict_raster.py`'s reader threads (one handle per
thread, reused across all its blocks — do NOT reopen per block, see above)
rather than pre-downloading/mosaicking 13 GB of raw UTM31N/32N tiles to disk
first. **Next step:** benchmark `--readers N > 1` for source.coop the same
way the local-tile case was benchmarked (1/3/6 readers, measuring aggregate
px/s) to see if it parallelizes as well as decompression-bound local reads
did, or whether network/GEE-side limits cap it lower.
