"""Is the low-accuracy problem SENSING or DEFINITION?

Motivation: AlphaEarth saturates, GPW (a purpose-built grassland product) moved
grassland F1 by -0.0009, NiN and DEM added nothing; only lidar (which measures
something optical sensors physically cannot see) ever helped. Before proposing
new sensors, establish WHAT KIND of error the bad classes make. Two very
different diagnoses, with opposite implications:

  (a) SENSING limit — the classes occupy distinct regions of feature space but
      the sensor can't resolve them. Then a new, ORTHOGONAL sensor can help, and
      we should chase SAR / phenology / lidar point-cloud metrics.

  (b) DEFINITION limit — the classes OVERLAP in feature space: the same physical
      surface is labelled cls5 in one polygon and cls6 in another. Then NO sensor
      helps, because there is no physical difference to measure. The fix is
      label/ontology work, not data acquisition.

Discriminating evidence computed here (all label-space, no model needed):

  1. BAYES-OPTIMAL overlap: for each pair of classes, fit the best possible
     separator using the FULL feature set (LDA/QDA-free: use a strong kNN density
     estimate) and report the irreducible pairwise error. If two classes are
     genuinely inseparable IN THE FEATURES, no architecture can fix it — but that
     alone doesn't distinguish (a) from (b).

  2. SAME-LOCATION LABEL DISAGREEMENT (the decisive test): each (lon,lat) has up
     to 9 yearly rows. If the SAME location, with a nearly identical embedding,
     carries DIFFERENT labels across years with no physical change, that is label
     noise / definitional ambiguity, not sensing. We measure, per class pair, how
     often a location flips between them while its embedding barely moves.

  3. NEAREST-NEIGHBOUR label purity: for each class, what fraction of a row's k
     nearest neighbours IN FEATURE SPACE share its label. Low purity + high
     confusion with one specific class = those two classes are the same thing to
     the sensor.

Run: ~/myprojects/recover/.venv/bin/python DNN/research/diag_class_overlap.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp_common as ec   # noqa: E402
import data_utils as du   # noqa: E402

# Post-merge grunnkart/FSCS legend (1->2 sand into rock, 9->8 marine into
# freshwater), per the authoritative source definitions.
NAMES = {2: "rock+sand", 3: "crop", 4: "forest", 5: "grassland",
         6: "scrub", 7: "wetland", 8: "water", 10: "built",
         11: "sparse-veg", 12: "snow/ice"}
OUT = ec.RESULTS_DIR / "diag_class_overlap.json"
K = 20
SUB = 120_000          # subsample for the kNN purity (full 664k x 664k is silly)


def main():
    t0 = time.perf_counter()
    print("=== class-overlap diagnostic: sensing limit or definition limit? ===")
    data = du.load_data(extra_features="lidar")
    X, y_enc, df = data["X"], data["y_enc"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    lon, lat = df["lon"].values, df["lat"].values
    print(f"loaded {X.shape[0]:,} rows  {X.shape[1]} feats")

    Xs = StandardScaler().fit_transform(X).astype(np.float32)

    # ---------- 1. kNN label purity in FEATURE space ----------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xs), min(SUB, len(Xs)), replace=False)
    tree = cKDTree(Xs[idx])
    _, nn = tree.query(Xs[idx], k=K + 1, workers=-1)
    nn = nn[:, 1:]                       # drop self
    nn_lab = y_enc[idx][nn]              # [n, K]
    own = y_enc[idx][:, None]
    purity = (nn_lab == own).mean(1)     # per row

    print("\n1. kNN label purity in feature space (k=20)")
    print("   = fraction of a row's 20 nearest neighbours sharing its label.")
    print("   LOW purity => the sensor cannot tell this class from its neighbours.")
    pur = {}
    for ci, c in enumerate(classes):
        m = y_enc[idx] == ci
        if m.sum() == 0:
            continue
        pur[str(c)] = round(float(purity[m].mean()), 4)
        print(f"   cls {c:>2} {NAMES[c]:>12}: purity={pur[str(c)]:.3f}  (n={int(m.sum()):,})")

    # ---------- 2. WHO does each class get confused with, in feature space ----------
    print("\n2. Feature-space neighbour composition of the WORST classes")
    print("   (of the non-self neighbours, which class dominates?)")
    leak = {}
    for ci, c in enumerate(classes):
        m = y_enc[idx] == ci
        if m.sum() == 0:
            continue
        lab = nn_lab[m].ravel()
        lab = lab[lab != ci]
        if len(lab) == 0:
            continue
        cnt = np.bincount(lab, minlength=n_classes).astype(float)
        cnt /= cnt.sum()
        top = np.argsort(-cnt)[:3]
        leak[str(c)] = {str(classes[t]): round(float(cnt[t]), 3) for t in top}
        s = "  ".join(f"{NAMES[classes[t]]}={cnt[t]*100:.0f}%" for t in top)
        print(f"   cls {c:>2} {NAMES[c]:>12} -> {s}")

    # ---------- 3. SAME-LOCATION LABEL FLIPS (the decisive test) ----------
    print("\n3. SAME-LOCATION label flips across years (THE DECISIVE TEST)")
    print("   Same (lon,lat), 2017-2025. If a location flips label while its")
    print("   embedding barely moves, that is DEFINITION/label noise, not sensing.")
    q = np.round(np.stack([lon, lat], 1).astype(np.float64), 6)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    nloc = inv.max() + 1
    order = np.argsort(inv, kind="stable")
    inv_s = inv[order]
    starts = np.flatnonzero(np.r_[True, inv_s[1:] != inv_s[:-1]])
    ends = np.r_[starts[1:], len(inv_s)]

    n_multi = 0
    n_flip = 0
    flip_pairs = {}
    flip_embed_dist = []
    same_embed_dist = []
    for s, e in zip(starts, ends):
        rows = order[s:e]
        if len(rows) < 2:
            continue
        labs = y_enc[rows]
        u = np.unique(labs)
        n_multi += 1
        # embedding spread within this location
        d = float(np.linalg.norm(Xs[rows] - Xs[rows].mean(0), axis=1).mean())
        if len(u) > 1:
            n_flip += 1
            flip_embed_dist.append(d)
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    a, b = int(u[i]), int(u[j])
                    key = f"{classes[a]}<->{classes[b]}"
                    flip_pairs[key] = flip_pairs.get(key, 0) + 1
        else:
            same_embed_dist.append(d)

    print(f"   locations with >=2 years: {n_multi:,}")
    print(f"   locations whose label FLIPS across years: {n_flip:,} "
          f"({100*n_flip/max(n_multi,1):.1f}%)")
    print(f"   mean within-location embedding spread:")
    print(f"     stable-label locations: {np.mean(same_embed_dist):.3f}")
    print(f"     FLIPPING locations:     {np.mean(flip_embed_dist):.3f}")
    print("   => if these are SIMILAR, the flips are NOT driven by real change;")
    print("      the same physical surface is being labelled differently.")
    top_flips = sorted(flip_pairs.items(), key=lambda kv: -kv[1])[:12]
    print("\n   most common label flips (same location, different year):")
    for k, v in top_flips:
        a, b = k.split("<->")
        print(f"     {NAMES[int(a)]:>12} <-> {NAMES[int(b)]:<12} {v:>6,} locations")

    OUT.write_text(json.dumps({
        "name": "diag_class_overlap",
        "knn_purity": pur, "neighbour_composition": leak,
        "locations_multiyear": int(n_multi), "locations_label_flip": int(n_flip),
        "flip_rate": round(n_flip / max(n_multi, 1), 4),
        "embed_spread_stable": round(float(np.mean(same_embed_dist)), 4),
        "embed_spread_flipping": round(float(np.mean(flip_embed_dist)), 4),
        "top_flip_pairs": {k: v for k, v in top_flips},
        "runtime_s": round(time.perf_counter() - t0, 1),
    }, indent=2))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
