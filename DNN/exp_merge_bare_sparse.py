"""Does merging sparse-veg (11) into bare ground (2) help or hurt?

The ask is a model whose bare-ground class also covers sparse vegetation. That
is a change to the ONTOLOGY, not to the recipe, so the only honest way to judge
it is a paired run: same architecture, same config, same folds, same relabel
protocol, one variable — the label space.

Three numbers come out, and they answer different questions:

  A. BASELINE (10 classes)         — per-class F1 as deployed today.
  A'. BASELINE, MERGED POST-HOC    — take A's out-of-fold predictions and map
      both truth and prediction 11 -> 2, then score on the 9-class space. This
      is what you get for FREE, by editing the legend of the existing model.
      No retraining, no new artifact.
  B. RETRAINED (9 classes)         — the model actually fit on merged labels.

B vs A' is the question that matters. If B does not beat A', retraining bought
nothing that a lookup table would not have given, and the 10-class model should
be kept (it strictly carries more information — you can always collapse it
later, but you cannot un-collapse B).

Macro-F1 across A and B is NOT comparable: it averages over 10 classes vs 9, and
dropping a hard class raises the mean for free. Only A' vs B is comparable, and
per-class F1 on the 8 untouched classes is comparable throughout. The script
prints all three and refuses to headline the incomparable one.

Also reported, because it decides whether the merge is even motivated: how much
of class 11's error mass actually goes to class 2 in the baseline. Merging two
classes only recovers the confusion BETWEEN them; if 11 is mostly confused with
6 (scrub), merging it into 2 recovers nothing and just discards a distinction.

Run:
  ~/myprojects/recover/.venv/bin/python DNN/exp_merge_bare_sparse.py
  ARCH=moe_shared N_EXPERTS=8 TOP_K=2 ~/…/python DNN/exp_merge_bare_sparse.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_utils as du          # noqa: E402
import dnn_core as C             # noqa: E402
from dnn_paths import result_path  # noqa: E402

# The authoritative codebook (DNN/confusion_matrix.py). Several older analysis
# scripts and the README carry a DIFFERENT, wrong legend (2="sparse-veg",
# 10="bare"); the lidar signatures settle it — class 2 sits at 13 m median
# elevation (coastal) and class 11 at 797 m (alpine).
NAMES = {2: "rock+sand", 3: "crop", 4: "forest", 5: "grassland", 6: "scrub",
         7: "wetland", 8: "water", 10: "built", 11: "sparse-veg", 12: "snow/ice"}
SRC, DST = 11, 2                      # sparse-veg -> bare ground
OUT = result_path("merge_bare_sparse.json")


def run_cv(tag, merge_extra, cfg):
    """3-fold spatial CV, returning per-class F1 and the OOF predictions.

    Mirrors dnn_core.cv_evaluate (same folds, same train-only cls12 centroid
    clean, test labels never touched) but keeps the out-of-fold predictions so
    the post-hoc merge and the confusion matrix can be computed from them.
    """
    du.MERGE_EXTRA = merge_extra          # merge_map() reads this at call time
    data = C.load_cached("lidar")
    X, y_enc, groups = data["X"], data["y_enc"], data["groups"]
    lon, lat = data["lon"], data["lat"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12 = classes.index(12) if 12 in classes else -1
    print(f"\n=== {tag}: {n_classes} classes {classes}  merge={du.merge_sig()}",
          flush=True)
    C.set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    oof = np.full(len(y_enc), -1, dtype=np.int64)
    t0 = time.perf_counter()
    for k, tr, te in du.fold_indices(y_enc, groups):
        ft = time.perf_counter()
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], None, cls12,
                                         lon[tr], lat[tr])
        tr_use = tr[keep]
        ens = C.fit_ensemble(X[tr_use], y_enc[tr_use], n_classes, cfg,
                             data["feat_cols"], classes, data.get("lidar_med"), rng)
        oof[te] = ens.predict_proba(X[te]).argmax(1)
        print(f"  fold {k}: F1={du.macro_f1(y_enc[te], oof[te], n_classes):.4f}  "
              f"dropped={int((~keep).sum())}  {time.perf_counter()-ft:.1f}s", flush=True)
    assert (oof >= 0).all(), "every row must get an out-of-fold prediction"
    return {
        "tag": tag, "classes": classes, "y_enc": y_enc, "oof": oof,
        "macro_f1": du.macro_f1(y_enc, oof, n_classes),
        "per_class": du.per_class_f1(y_enc, oof, n_classes),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def collapse(y_enc, classes, src, dst):
    """Re-express encoded labels on the label space with `src` folded into `dst`."""
    vals = np.asarray(classes)[y_enc]
    vals[vals == src] = dst
    new_classes = sorted(set(classes) - {src})
    remap = {c: i for i, c in enumerate(new_classes)}
    return np.array([remap[v] for v in vals], dtype=np.int64), new_classes


def show(title, classes, per_class, macro, note=""):
    print(f"\n{title}   macro-F1 = {macro:.4f} {note}")
    for c, f in zip(classes, per_class):
        print(f"    {c:>3} {NAMES.get(c,'?'):<11} {f:.4f}")


def main():
    cfg = C.Config()
    print(f"device={C.DEVICE}  arch={cfg.arch}  ensemble={cfg.n_ensemble}  "
          f"merging {SRC} ({NAMES[SRC]}) -> {DST} ({NAMES[DST]})", flush=True)

    A = run_cv("A baseline (10-class)", "", cfg)
    B = run_cv(f"B retrained ({SRC}->{DST})", f"{SRC}:{DST}", cfg)

    # A': collapse A's truth AND predictions, then score on B's label space.
    yA, clsA = collapse(A["y_enc"], A["classes"], SRC, DST)
    pA, _ = collapse(A["oof"], A["classes"], SRC, DST)
    a_macro = du.macro_f1(yA, pA, len(clsA))
    a_pc = du.per_class_f1(yA, pA, len(clsA))
    assert clsA == B["classes"], "post-hoc merge must land on B's label space"

    show("A  baseline, 10 classes", A["classes"], A["per_class"], A["macro_f1"],
         "(NOT comparable to B — different class count)")
    show("A' baseline collapsed post-hoc (no retraining)", clsA, a_pc, a_macro)
    show("B  retrained on merged labels", B["classes"], B["per_class"], B["macro_f1"])

    print("\n--- B vs A' (the comparable pair) ---")
    print(f"    {'class':<16} {'A- post-hoc':>12} {'B retrained':>12} {'delta':>8}")
    for i, c in enumerate(clsA):
        d = B["per_class"][i] - a_pc[i]
        flag = "  <-- worse" if d < -0.005 else ("  <-- better" if d > 0.005 else "")
        print(f"    {c:>3} {NAMES.get(c,'?'):<12} {a_pc[i]:>12.4f} "
              f"{B['per_class'][i]:>12.4f} {d:>+8.4f}{flag}")
    print(f"    {'MACRO':<16} {a_macro:>12.4f} {B['macro_f1']:>12.4f} "
          f"{B['macro_f1']-a_macro:>+8.4f}")

    # Is the merge motivated? Where does class SRC's error actually go?
    iS, iD = A["classes"].index(SRC), A["classes"].index(DST)
    mS = A["y_enc"] == iS
    wrong = mS & (A["oof"] != iS)
    dest = np.bincount(A["oof"][wrong], minlength=len(A["classes"]))
    print(f"\n--- where does true {SRC} ({NAMES[SRC]}) go wrong in the baseline? "
          f"({wrong.sum():,} of {mS.sum():,} rows) ---")
    for i in np.argsort(dest)[::-1][:4]:
        c = A["classes"][i]
        print(f"    -> {c:>3} {NAMES.get(c,'?'):<12} {dest[i]:>7,}  "
              f"{100*dest[i]/max(wrong.sum(),1):>5.1f}% of its errors")
    recov = 100 * dest[iD] / max(wrong.sum(), 1)
    # and the reverse direction
    mD = A["y_enc"] == iD
    wrongD = mD & (A["oof"] != iD)
    destD = np.bincount(A["oof"][wrongD], minlength=len(A["classes"]))
    print(f"    reverse: true {DST} misread as {SRC}: {destD[iS]:,} "
          f"({100*destD[iS]/max(wrongD.sum(),1):.1f}% of class-{DST} errors)")
    print(f"\n    only {recov:.1f}% of class-{SRC} errors are recovered by the merge; "
          f"the rest survive it.")

    res = {
        "kind": "merge_bare_sparse", "merge": f"{SRC}->{DST}",
        "names": {str(k): v for k, v in NAMES.items()},
        "config": cfg.to_dict(),
        "A_baseline": {"classes": A["classes"], "macro_f1": round(A["macro_f1"], 4),
                       "per_class": {str(c): round(float(v), 4)
                                     for c, v in zip(A["classes"], A["per_class"])},
                       "wall_s": A["wall_s"]},
        "A_posthoc_merged": {"classes": clsA, "macro_f1": round(a_macro, 4),
                             "per_class": {str(c): round(float(v), 4)
                                           for c, v in zip(clsA, a_pc)}},
        "B_retrained": {"classes": B["classes"], "macro_f1": round(B["macro_f1"], 4),
                        "per_class": {str(c): round(float(v), 4)
                                      for c, v in zip(B["classes"], B["per_class"])},
                        "wall_s": B["wall_s"]},
        "B_minus_A_posthoc": {str(c): round(float(B["per_class"][i] - a_pc[i]), 4)
                              for i, c in enumerate(clsA)},
        f"class{SRC}_error_destinations": {
            str(A["classes"][i]): int(dest[i]) for i in range(len(A["classes"]))},
        f"class{SRC}_errors_recovered_by_merge_pct": round(recov, 1),
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(res, indent=2))
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
