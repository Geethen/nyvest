"""Change / no-change raster from two classified years, plus the diagnostics
needed to know how much of it to believe.

Output `change_<y0>_<y1>.tif`, uint8:
    0  nodata   (either year unclassified — kept as the file's nodata value)
    1  no change
    2  change

The comparison itself is trivial — the two maps share a grid pixel for pixel —
and that is exactly why this script spends most of its effort elsewhere. A raw
class-flip map over two independent classifications is dominated by model
variance, not by change on the ground, and nothing about the raster says so. So
the same single pass also computes:

  * the full transition matrix (which flips, and how big);
  * the flip rate restricted to pixels where BOTH years produced a SINGLETON
    conformal prediction set. Those are the pixels the calibrated model was
    confident about twice, so a flip there is much harder to explain as noise.
    (The stricter screen — disjoint prediction sets — needs the 9-band inset
    rasters, ~60 GB of decompressed reads for the pair, versus ~7 GB for the
    1-band set-size rasters. The singleton test gets most of the signal for a
    tenth of the I/O.)
  * the flip rate for classes whose true change is ~zero over six years.
    Built-up land does not revert; whatever this reports for class 10 is a
    lower bound on the error rate of the whole product.

Run:
  PY DNN/make_change_layer.py --dir <folder with classified_*.tif> \
     --years 2018 2024 [--model models/dnn_final_moe8_merged.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

import rasterio                                            # noqa: E402
from rasterio.windows import Window                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

NODATA, NO_CHANGE, CHANGE = 0, 1, 2
# Classes whose real-world change over a few years is ~0 in this AOI. Used only
# as a diagnostic floor, never to mask anything.
IRREVERSIBLE = {10: "built"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--years", nargs=2, type=int, default=[2018, 2024])
    ap.add_argument("--model", default=None, help="for class labels in the report")
    ap.add_argument("--rows", type=int, default=4096, help="strip height per read")
    args = ap.parse_args()

    d = Path(args.dir)
    y0, y1 = args.years
    p0, p1 = d / f"classified_{y0}.tif", d / f"classified_{y1}.tif"
    s0, s1 = d / f"uq_{y0}_setsize.tif", d / f"uq_{y1}_setsize.tif"
    out = d / f"change_{y0}_{y1}.tif"
    have_ss = s0.exists() and s1.exists()

    labels = {}
    if args.model:
        import torch
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        try:
            from write_qgis_styles import resolve_classes
            labels = {c: n for c, _, n in resolve_classes(args.model)}
        except Exception:
            labels = {c: str(c) for c in ck["classes"]}

    t0 = time.perf_counter()
    with rasterio.open(p0) as a, rasterio.open(p1) as b:
        if (a.width, a.height, a.crs, a.transform) != (b.width, b.height, b.crs, b.transform):
            raise SystemExit(
                f"{p0.name} and {p1.name} are not on the same grid — a pixel-wise "
                f"change layer would compare different places. Reproject first.")
        prof = a.profile.copy()
        prof.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA,
                    compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, bigtiff="IF_SAFER")
        W, H = a.width, a.height
        # Matrix size comes from the label space, not from a sampled window: any
        # probe window can legitimately be all-nodata (most of this AOI's
        # bounding box is outside the counties), and sizing off that crashed.
        M = (max(labels) if labels else 12) + 1
        trans = np.zeros((M, M), dtype=np.int64)      # [from, to]
        n_valid = n_change = 0
        n_conf = n_conf_change = 0

        ssa = rasterio.open(s0) if have_ss else None
        ssb = rasterio.open(s1) if have_ss else None
        with rasterio.open(out, "w", **prof) as dst:
            for row in range(0, H, args.rows):
                h = min(args.rows, H - row)
                win = Window(0, row, W, h)
                A = a.read(1, window=win)
                B = b.read(1, window=win)
                valid = (A != 0) & (B != 0)
                chg = valid & (A != B)
                o = np.where(valid, np.where(chg, CHANGE, NO_CHANGE), NODATA).astype(np.uint8)
                dst.write(o, 1, window=win)
                n_valid += int(valid.sum())
                n_change += int(chg.sum())
                # transition matrix over valid pixels
                np.add.at(trans, (A[valid].astype(np.int64), B[valid].astype(np.int64)), 1)
                if have_ss:
                    SA = ssa.read(1, window=win)
                    SB = ssb.read(1, window=win)
                    conf = valid & (SA == 1) & (SB == 1)
                    n_conf += int(conf.sum())
                    n_conf_change += int((conf & chg).sum())
                if (row // args.rows) % 4 == 0:
                    print(f"  {row+h}/{H} rows  {n_valid/1e6:.0f}M valid  "
                          f"{100*n_change/max(n_valid,1):.2f}% changed", flush=True)
        if ssa: ssa.close()
        if ssb: ssb.close()

    el = time.perf_counter() - t0
    pct = 100 * n_change / max(n_valid, 1)
    print(f"\nwrote {out}  ({el:.0f}s)")
    print(f"  valid  {n_valid:,}")
    print(f"  change {n_change:,}  ({pct:.2f}%)   no change {n_valid-n_change:,}")

    rep = {"kind": "change_layer", "years": [y0, y1], "raster": str(out),
           "encoding": {"0": "nodata", "1": "no change", "2": "change"},
           "valid_px": n_valid, "change_px": n_change, "change_pct": round(pct, 3)}

    if have_ss:
        cpct = 100 * n_conf_change / max(n_conf, 1)
        print(f"\n  both years a SINGLETON conformal set: {n_conf:,} px "
              f"({100*n_conf/max(n_valid,1):.1f}% of valid)")
        print(f"    of those, changed: {n_conf_change:,}  ({cpct:.2f}%)")
        print(f"    -> confident-pixel flip rate is {cpct/max(pct,1e-9):.2f}x the overall rate")
        rep.update(confident_px=n_conf, confident_change_px=n_conf_change,
                   confident_change_pct=round(cpct, 3))

    print(f"\n  top transitions (% of all valid px):")
    flat = [(trans[i, j], i, j) for i in range(M) for j in range(M)
            if i != j and trans[i, j] > 0]
    for n, i, j in sorted(flat, reverse=True)[:10]:
        print(f"    {i:>3} {labels.get(i,'?'):<24} -> {j:>3} {labels.get(j,'?'):<24}"
              f" {n:>12,}  {100*n/n_valid:>5.2f}%")

    print(f"\n  error floor from classes that should not change:")
    for c, name in IRREVERSIBLE.items():
        if c >= M:
            continue
        tot = int(trans[c].sum())
        kept = int(trans[c, c])
        if tot:
            print(f"    {c} {name}: {tot-kept:,} of {tot:,} px left the class "
                  f"({100*(tot-kept)/tot:.1f}%) — real change here is ~0, so this "
                  f"is error")
            rep[f"class{c}_left_pct"] = round(100 * (tot - kept) / tot, 2)

    rep["transitions"] = {f"{i}->{j}": int(trans[i, j]) for i in range(M)
                          for j in range(M) if trans[i, j] > 0}
    (d / f"change_{y0}_{y1}.json").write_text(json.dumps(rep, indent=2))
    print(f"\n  report -> {d / f'change_{y0}_{y1}.json'}")


if __name__ == "__main__":
    main()
