"""Aggregate reports/results/lc_shard_*.json into the learning-curve outputs.
Averages seeds per fraction; plots per-class + macro learning curves."""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dnn_paths import RESULTS_DIR, figure_path, result_path  # noqa: E402

NAMES = {2: "2 sparse-veg", 3: "3 forest", 4: "4 forest", 5: "5 GRASSLAND",
         6: "6 open-upland", 7: "7 mire/wet", 8: "8 water", 10: "10 bare",
         11: "11 built/infra", 12: "12 snow/ice"}


def main():
    shards = [json.loads(Path(f).read_text()) for f in glob.glob(str(RESULTS_DIR / "lc_shard_*.json"))]
    if not shards:
        print("no shards found"); return
    byfrac = defaultdict(list)
    for s in shards:
        byfrac[s["frac"]].append(s)
    fracs = sorted(byfrac)
    classes = sorted(int(c) for c in shards[0]["per_class"])
    curve = {"fractions": fracs, "n_train": [], "macro": [],
             "per_class": {str(c): [] for c in classes}}
    for fr in fracs:
        grp = byfrac[fr]
        curve["n_train"].append(int(np.mean([g["n_train"] for g in grp])))
        curve["macro"].append(round(float(np.mean([g["macro"] for g in grp])), 4))
        for c in classes:
            vals = [g["per_class"][str(c)] for g in grp]
            curve["per_class"][str(c)].append(round(float(np.mean(vals)), 4))
    result_path("learning_curve.json").write_text(json.dumps(curve, indent=2))
    print("fractions:", fracs)
    print("n_train:  ", curve["n_train"])
    print("macro:    ", curve["macro"])
    for c in classes:
        print(f"  class {c:2d} {NAMES.get(c,''):14s}: {curve['per_class'][str(c)]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = np.array(curve["n_train"])
        fig, ax = plt.subplots(figsize=(9, 6))
        for c in classes:
            ax.plot(n, curve["per_class"][str(c)], marker="o", label=NAMES.get(c, str(c)))
        ax.plot(n, curve["macro"], "k--", lw=2.5, marker="s", label="MACRO")
        ax.set_xscale("log")
        ax.set_xlabel("train rows (stratified, log scale)")
        ax.set_ylabel("test F1 (fold 0)")
        ax.set_title("Class-wise learning curves — rising=wants more data, flat=confusion-bound")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        png = figure_path("learning_curve.png")
        fig.savefig(png, dpi=130)
        print(f"saved -> {png}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
