# Timber spike — decision: NO-GO for multiclass (2026-06-22)

Goal: use `timber-compiler` (https://github.com/kossisoroyce/timber, AOT tree→C99
compiler, ~336× single-sample speedup) to accelerate the CatBoost dense raster pass
and the live AL loop.

## What works
- `timber-compiler==0.6.0` installs cleanly; gcc 13.3 present for the C build.
- It DOES expose an in-process Python API (docs only mention an HTTP server):
  `from timber.runtime import TimberPredictor` → `from_model(json)` / `from_artifact(dir)`
  → `predict(X) -> (n, n_outputs)` via ctypes over a compiled `.so`. Batch-capable.
  So HTTP is not required — good.

## Blocker (hard NO-GO for our use)
The CatBoost front-end is **single-output only**; multiclass is declared but not implemented.
On our real model (10 classes, `loss_function="MultiClass"`):

1. `frontends/catboost_parser.py:68` crashes:
   `scale = float(bias_data[0][0])` — but multiclass `scale_and_bias = [1, [0,0,…0]]`
   (scalar scale + length-10 bias vector), so `bias_data[0]` is the int `1` →
   `TypeError: 'int' object is not subscriptable`.
2. Deeper issue even if (1) were patched: CatBoost multiclass stores `leaf_values` as
   **n_leaves × n_classes** (confirmed: a depth-4 tree has 16 leaves and 160 leaf values
   = 16×10). Timber's IR (`Tree`/`TreeNode`) holds a single scalar `leaf_value` per leaf
   and the C codegen accumulates one output — there is no representation for a per-class
   leaf vector. Supporting multiclass = extending the IR to vector leaves + rewriting the
   C codegen to emit per-class accumulation + softmax. That's a fork of the library, not
   a config change.

## Follow-up (2026-06-22, after "timber will make inference near-instant if set up correctly")
Tested the obvious workaround — **binary one-vs-rest** (scalar-leaf, so timber-representable)
+ a JSON patch to fix the `scale_and_bias` crash (CatBoost writes `[1,[0]]`; rewrite to
`[[1.0],[0.0]]` so the parser's `[0][0]`/`[1][0]` indexing works). Two hard results, both
negative for OUR use:

1. **Accuracy is wrong.** Timber compiles + runs the binary model, but its output only
   *approximately* tracks CatBoost: `corr(timber, raw_margin)=0.977` yet **max abs error
   8.7 in log-odds**, and `max|sigmoid(timber) − P(class=1)| = 0.50` — probabilities off by
   up to a coin flip. Almost certainly its oblivious-tree leaf indexing
   (`catboost_parser._level_of_node` / `split_idx`) doesn't match CatBoost's symmetric-tree
   leaf order. A wrong map, not a slow one.
2. **Timber is SLOWER for our access pattern.** Batched predict of 3000 rows: timber 14.2 ms
   vs CatBoost 2.7 ms (**0.2×**). The ~336× headline is *single-sample* latency (1 row, ~2 µs).
   Dense raster prediction and the AL rescoring loop are **batched over ~1M / ~75k rows**,
   where CatBoost's vectorised C++ already wins. The 336× regime does not apply to us.

So timber does not help here even via OvR: it would need the maintainer to (a) fix the
multiclass + binary `scale_and_bias` parse, AND (b) fix the leaf-indexing accuracy bug,
AND we'd still only benefit in a single-sample-latency regime we don't operate in.

## Decision
Use **native CatBoost `predict_proba`** for both the dense single-tile raster and the live
loop. Measured: 954k px/tile in **5.6 s** (~170k px/s); fast retrain ~11 s; live rescore of
75k locations ~0.1 s. `predict_raster.py` keeps a pluggable `--backend {catboost,timber}`
seam so timber can be revisited IF upstream fixes both bugs — but it only pays off for
single-row latency, which is not this app's workload.

Reproduce: train any `MultiClass` CatBoost, `save_model(format="json")`,
`TimberPredictor.from_model(path, format_hint="catboost")` → raises at parser line 68.
