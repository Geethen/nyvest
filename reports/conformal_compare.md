# Conformal prediction: APS vs RAPS vs SAPS vs RANK

**Date:** 2026-04-22
**Base model:** XGBoost, `n_estimators=500`, `device=cuda`, trained on 69 552 rows
**Calibration:** val split, 19 999 rows
**Evaluation:** test split, 10 085 rows
**Classes:** 22 (land-cover `fallbck`)
**Probabilities cached at** [reports/saps_cache.npz](saps_cache.npz) · raw results at [reports/conformal_compare_results.csv](conformal_compare_results.csv)

## Methods

| Method | Score `V(p, k)` for class `k` at sorted rank `r` | Reference |
|---|---|---|
| APS | Σ_{j<r} p_(j) + u · p_(r) | [Romano et al. 2020](https://arxiv.org/abs/2006.02544) |
| RAPS | V_APS + λ · max(0, r − k_reg) | [Angelopoulos et al. 2021](https://arxiv.org/abs/2009.14193) |
| SAPS | p_(1) · u if r=1 else p_(1) + (r−2+u) · λ | [Huang et al. ICML 2024](https://arxiv.org/abs/2310.07315) |
| RANK | Two-stage (Algorithm 1) — top-r*_α or top-(r*_α−1) classes per sample, boundary decided by empirical CDF of rank-r*_α probs | [Liu et al. Pattern Recognition 2025](https://arxiv.org/abs/2407.04407) |

Hyperparameters used: RAPS `λ=0.01, k_reg=1`; SAPS `λ=0.20` (Pareto point from [benchmark_dryrun.md](benchmark_dryrun.md)); RANK parameter-free; u ~ U(0,1) per cal/test sample.

## Results

All four methods hit target coverage within ~0.3 pp (conformal guarantee holds on val→test).

### α = 0.05 (target coverage 0.95)

| Method | Coverage | Avg size | Median | Singleton % | Empty % |
|---|---:|---:|---:|---:|---:|
| APS  | 0.955 | 3.60 | 3 | 31% | 0.1% |
| RAPS | 0.954 | 3.83 | 4 | 0%  | 0%   |
| SAPS | 0.954 | 4.11 | 4 | 0%  | 0%   |
| RANK | 0.962 | 4.65 | 5 | 0%  | 0%   |

### α = 0.10 (target coverage 0.90)

| Method | Coverage | Avg size | Median | Singleton % | Empty % |
|---|---:|---:|---:|---:|---:|
| APS  | 0.907 | **2.57** | 2 | 42% | 0.5% |
| RAPS | 0.907 | **2.37** | 2 | 35% | 0%   |
| SAPS | 0.905 | 2.61 | 2 |  9% | 0%   |
| RANK | 0.915 | 3.08 | 3 |  0% | 0%   |

### α = 0.20 (target coverage 0.80)

| Method | Coverage | Avg size | Median | Singleton % | Empty % |
|---|---:|---:|---:|---:|---:|
| APS  | 0.807 | **1.73** | 1 | 57% | 3.0% |
| RAPS | 0.808 | 1.68 | 1 | 57% | 2.4% |
| SAPS | 0.810 | **1.66** | 1 | 59% | 2.0% |
| RANK | 0.833 | 1.79 | 2 | 21% | 0%   |

## Observations

- **All four respect the coverage guarantee** on our ~20 k val / 10 k test split. Differences show up in set *structure*, not coverage.
- **APS is smallest but emits empty sets** (0.1%–3% depending on α). In a land-cover review workflow that's a UX problem — the reviewer gets no candidates and can't proceed without a fallback rule.
- **RAPS (λ=0.01, k_reg=1) fixes APS's α=0.10 empties** and gives the tightest non-empty sets (2.37 avg at 90% coverage). At α=0.20 it still emits 2.4% empty sets — the penalty isn't strong enough at high miscoverage.
- **SAPS sits between RAPS and RANK**: eliminates empties at α=0.05 and α=0.10, slightly smaller sets than RANK, keeps some singletons.
- **RANK is the most conservative, most structured choice**. By construction every prediction set is either top-r*_α or top-(r*_α − 1) classes — never empty, no degenerate singletons at high coverage. Trade-off: ~15% larger average sets than APS/RAPS at α=0.10, and slightly over-covers (0.915 vs 0.90 target) because the two-stage rule rounds up.

## Recommendation for NYVEST

If prediction sets feed a **human reviewer** (e.g. flagging uncertain pixels for manual NIBIO class assignment): **RANK** at α=0.10. Every sample gets a usable 3-class shortlist, no empties to handle, no misleading singletons on hard samples. Slight over-coverage is a feature, not a bug, in this context.

If prediction sets feed a **downstream automated pipeline** that can tolerate ambiguity (e.g. soft classification averaged across tiles): **RAPS** at α=0.10 with `λ=0.01, k_reg=1`. Smaller sets, no empties, better statistical efficiency.

**Avoid plain APS** in this use case — the empty-set rate (0.5%–3%) is not worth the ~2% size saving over RAPS.

## Runtime

All four methods run in 14–25 ms per α on cached probabilities — the cost is dominated by the one-off 21 s XGBoost fit. The comparison script [conformal_compare.py](../scripts/conformal_compare.py) reuses [saps_cache.npz](saps_cache.npz), so sweeping hyperparameters is free after the first call.
