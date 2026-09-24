# Locality round — complete

**25 records, 0 wins, 22 ties, 2 splits, 1 bound.** Both rounds finished; the page,
`DNN/README.md` and findings.json are current. Nothing here is left running.

**Closed out afterwards** (two questions the round left unanswered):

- **A floor.** `probe_linear` 0.7040 and `rf_tuned` 0.6945 against the deployed
  0.7414, on identical folds — so the net's margin over a *linear* read of the
  same 67 features is +0.0373 and this round's best arm adds +0.0040 on top,
  9× smaller. The forest is the sharper result: inner val **0.8825**, held-out
  0.6945, a val-to-test gap of 0.188 against the net's 0.056, with every fold
  choosing the least-regularised grid corner. Same lesson as `ctrl_capacity`,
  from a fourth direction.
- **A price.** `bench_inference.py` puts every arm on the deployed inference
  path. The baseline uses 2.7 min of GPU behind a ~15 min raster read; the best
  arm (`moe_shared@8`) uses 13.5 — hidden by the read on this box, but 87% of the
  headroom gone — and `@16` crosses it. `mod_depth`, the one open thread, costs
  1.8×. Top-k buys capacity, not FLOPs: the forward loops over every expert.
  **No arm here earns its inference cost.**

## Files

| file | what it is |
|---|---|
| `moe_layers.py` | the mechanisms + 14 pre-flight self-tests (`python moe_layers.py`) |
| `trials_local.py` | the 13 trials, each with a hypothesis recorded before it ran |
| `loop_local.py` | the round's scheduler and its pre-registered sweeps |
| `diag_locality.py` | what a local label is worth — the label-budget study |
| `backfill_val.py` | recovers inner-val F1 into records written before it was stored |
| `fix_verdicts.py` | re-derives verdicts from stored numbers (see the bug note below) |
| `smoke_local.py` | integration smoke test: builders + both fold overrides at 2 epochs |
| `trials_classic.py` | the classical floor — linear probe + tuned RF, 13 self-tests |
| `bench_inference.py` | what each arm costs on the deployed path, 20 self-tests |

## Re-running anything

```bash
cd DNN/autoresearch
~/myprojects/recover/.venv/bin/python run.py <trial>       # one trial
~/myprojects/recover/.venv/bin/python -u loop_local.py --workers 2   # resumable
~/myprojects/recover/.venv/bin/python build_artifact.py    # regenerate report.html

# the two additions below (CPU-only ~35 min, and GPU ~4 min)
~/myprojects/recover/.venv/bin/python run.py probe_linear
~/myprojects/recover/.venv/bin/python run.py rf_tuned
~/myprojects/recover/.venv/bin/python bench_inference.py --cpu
```

Pre-flight self-tests, all offline and seconds each:

```bash
~/myprojects/recover/.venv/bin/python moe_layers.py            # 14
~/myprojects/recover/.venv/bin/python trials_classic.py        # 13
~/myprojects/recover/.venv/bin/python bench_inference.py --selftest   # 20
```

## Two traps this round hit, both fixed

1. **Diagnostics share `results/` with trial records** but have no `name`/`f1_mean`.
   `loop.py:_save_state()` and `build_artifact.py:load()` both crashed on
   `diag_locality.json`, and the scheduler died *between* trials — silently, so it
   looked like a slow GPU. Both now filter on record shape. If trials stop
   launching, read `logs/loop_local.log` for a traceback before assuming anything.
2. **The verdict rule mislabelled positive results.** `WIN → tie → LOSS` sent any
   mean above +0.003 that failed the all-folds test to `LOSS`; +0.0040 and +0.0033
   were published as losses. `ar_common.py` now emits `split`. A loop process holds
   the old module until it restarts, so run `fix_verdicts.py` after any run that
   started before the fix — it relabels from stored numbers and re-runs nothing.

## The one open thread

`mod_depth` is the only arm the matched-capacity control does not explain: +0.0028
(reseed +0.0013), positive on all 6 fold-by-seed combinations, all four weak
classes up, and it scores *lower* on inner val than the matched-capacity plain MLP
while scoring *higher* on the held-out region. Its capacity sweep has a symmetric
interior optimum at 0.5 (0.25 → +0.0017, 0.50 → +0.0028, 0.75 → +0.0017).

Worth exactly one more experiment: **a 5-seed average of `mod_depth@capacity=0.5`**,
not another single draw. Margins here sit near the 0.0013 noise floor.

It is also the cheapest of the three arms that moved anything — 1.8× the deployed
forward pass, against 5.0× for `moe_shared@8` — so if the 5-seed average holds it
is the only one that could be adopted without eating the I/O headroom. If it does
not hold, the floor result says where the remaining headroom is not: a linear
model already reaches 0.7040, and no estimator change has beaten the noise floor
in three rounds.
