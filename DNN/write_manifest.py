"""Write a single manifest describing all inference-pipeline output rasters.

The pipeline emits several rasters with different dtypes, band layouts, class
codes, a scaled-proba encoding, and per-file nodata values. Rather than make the
end user reverse-engineer that from the GeoTIFF tags, this writes one companion
`<name>.json` (+ a readable `<name>.md`) documenting every output found in a
directory: its role, bands, class-code legend, encoding/scale, and nodata.

It inspects the files that actually exist (via rasterio) so the manifest matches
what was really produced — it does not assume a fixed set. Run it after
`predict_raster.py` (and optionally the lidar/coverage steps):

  PY DNN/write_manifest.py --dir /path/to/outputs --out /path/to/outputs/MANIFEST
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import rasterio

# Raw class codes the model emits -> human labels, from the authoritative
# grunnkart codebook (DNN/confusion_matrix.py, DNN/relabel_v2.py).
# Codes 1 and 9 never appear: training merges 1->2 and 9->8 (see DNN/README.md).
#
# Code 11 was previously labelled "infra" here, which is WRONG and shipped in
# the manifest: 11 is SPARSE VEGETATION (alpine — 797 m median elevation in the
# training points), while 10 is the built/settlement class. Several analysis
# scripts and DNN/README.md still carry that swapped legend; the codebook and
# the per-class lidar signatures both say otherwise.
CLASS_LABELS = {2: "rock+sand (bare ground)", 3: "crop", 4: "forest",
                5: "grassland", 6: "scrub", 7: "wetland", 8: "water",
                10: "built", 11: "sparse-veg", 12: "snow/ice"}

# Codes a run may FOLD AWAY via $MERGE_EXTRA (data_utils.merge_map). The
# manifest must describe the label space the raster actually uses, so the legend
# is filtered to the codes present and the merge is stated explicitly.
MERGED_LABELS = {(11, 2): "rock+sand + sparse-veg (bare ground incl. sparsely "
                          "vegetated)"}

# Role + human note per known output basename-suffix. Matched by endswith on the
# stem so it works regardless of the AOI/year prefix (classified_2024, uq_2024…).
ROLES = {
    "classified": "Predicted land-cover class map. Single band of RAW class "
                  "codes (see class_legend); nodata = unclassified "
                  "(outside AOI / no AlphaEarth data).",
    "_pcal": "Per-class calibrated probability (uncertainty). Each band is one "
             "class; divide by proba_scale to get a probability in [0,1]. Bands "
             "sum to ~1 per valid pixel. Calibration: Venn-Abers or temperature "
             "scaling, whichever fit_calibration.py found better.",
    "_setsize": "Conformal prediction-set size per pixel (0..n_classes): how "
                "many classes the LAC+Mondrian conformal predictor could not "
                "rule out at the calibrated risk level. 1 = confident, higher = "
                "more ambiguous.",
    "_inset": "Per-class conformal-set membership (0/1): 1 if the class is in "
              "the pixel's conformal prediction set. One band per class.",
    "lidar_3band": "Terrain features fed to the model (elevation m, TRI, canopy "
                   "height m). Bands: elevation, tri, tch. nodata = no lidar "
                   "(model uses training-set medians there at inference).",
    "lidar_coverage": "Lidar coverage mask: 1 = real lidar used, 0 = "
                      "median-filled at inference, 255 = outside AOI.",
    "aef_2024": "AlphaEarth embedding mosaic VRT (model input, 64 bands "
                "A00..A63). Not a deliverable — an intermediate.",
    "change_": "Class-flip map between two epochs: 0 nodata, 1 no change, 2 "
               "change. RAW flips of two independent classifications — most of "
               "it is model variance, not change on the ground. See the "
               "companion change_*.json for the measured error floor before "
               "using it for anything.",
    "nature_loss": "Natural -> anthropogenic conversion between two epochs, "
                   "graded by confidence. Band 1 loss_tier: 0 nodata, 1 no "
                   "loss, 2 candidate (fails the screens — mostly noise), 3 "
                   "passes the screens, 4 passes screens + minimum mapping "
                   "unit. USE TIER 4. Band 2 from_class: the natural class "
                   "lost (see class_legend). Band 3 screens: screen bitmask, "
                   "bit 6 (64) marks reverse-direction control pixels. "
                   "Grassland -> cropland is excluded by default. See the "
                   "companion nature_loss_*.json for the screen comparison and "
                   "the false-positive control.",
}


def _role(stem: str) -> str:
    for key, note in ROLES.items():
        if key.startswith("_"):
            if stem.endswith(key):
                return note
        elif stem.startswith(key) or key in stem:
            return note
    return "(undocumented output — inspect band descriptions)"


def resolve_legend(model_path):
    """(labels, note) for the label space this run actually produced.

    A manifest that lists classes the raster cannot contain is worse than none —
    it sends the reader looking for a sparse-veg class that was merged away — so
    the legend is filtered to the model's own class list when one is given.
    """
    labels = dict(CLASS_LABELS)
    note = ("Class codes 1 and 9 are absent by design: training merges "
            "1->2 and 9->8 (see DNN/README.md). Codes are RAW, not 0..N.")
    if not model_path:
        return labels, note
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    present = list(ck["classes"])
    for (src, dst), lab in MERGED_LABELS.items():
        if dst in present and src not in present:
            labels[dst] = lab
            note += (f" This run additionally merged {src} -> {dst}, so code "
                     f"{src} is absent and code {dst} covers both.")
    labels = {k: v for k, v in labels.items() if k in present}
    note += f" Legend taken from {os.path.basename(model_path)}."
    return labels, note


def describe(path: str, labels: dict) -> dict:
    with rasterio.open(path) as s:
        stem = os.path.splitext(os.path.basename(path))[0]
        entry = {
            "file": os.path.basename(path),
            "role": _role(stem),
            "driver": s.driver,
            "size": f"{s.width} x {s.height}",
            "crs": str(s.crs),
            "dtype": s.dtypes[0],
            "bands": s.count,
            "nodata": None if s.nodata is None else float(s.nodata),
            "band_descriptions": [d or f"band{i+1}"
                                  for i, d in enumerate(s.descriptions)],
            "compression": (s.compression.value if s.compression else None),
        }
    # attach scale + legend where they apply
    if stem.endswith("_pcal"):
        entry["proba_scale"] = 60000
        entry["decode"] = "probability = pixel_value / proba_scale"
    if "classified" in stem or stem.endswith("_pcal") or stem.endswith("_inset"):
        entry["class_legend"] = {str(k): v for k, v in labels.items()}
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of output rasters")
    ap.add_argument("--out", required=True,
                    help="manifest path stem (writes <out>.json + <out>.md)")
    ap.add_argument("--model", default=None,
                    help="the .pt used for inference; its class list decides "
                         "which legend entries apply. Without it the manifest "
                         "documents the full 10-class legend, which is WRONG "
                         "for a run that merged classes (e.g. MERGE_EXTRA=11:2)")
    ap.add_argument("--glob", default="*.tif",
                    help="which files to describe (default *.tif; VRTs skipped "
                         "unless matched)")
    args = ap.parse_args()

    labels, note = resolve_legend(args.model)

    import glob as globmod
    paths = sorted(globmod.glob(os.path.join(args.dir, args.glob)))
    paths = [p for p in paths if not p.endswith(".aux.xml")]
    entries = []
    for p in paths:
        try:
            entries.append(describe(p, labels))
        except rasterio.errors.RasterioIOError:
            continue

    manifest = {
        "generated": date.today().isoformat(),
        "dir": os.path.abspath(args.dir),
        "model": os.path.abspath(args.model) if args.model else None,
        "class_legend": {str(k): v for k, v in labels.items()},
        "note": note,
        "outputs": entries,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

    # readable markdown
    md = [f"# Inference outputs — {manifest['dir']}",
          f"\nGenerated {manifest['generated']}.\n",
          "## Class legend (raw codes)\n",
          "| code | class |", "|---|---|"]
    for k, v in labels.items():
        md.append(f"| {k} | {v} |")
    md.append(f"\n_{manifest['note']}_\n")
    md.append("## Files\n")
    for e in entries:
        md.append(f"### `{e['file']}`\n")
        md.append(f"{e['role']}\n")
        md.append(f"- **{e['bands']} band(s)**, {e['dtype']}, {e['size']}, "
                  f"nodata `{e['nodata']}`, {e['compression']} compression")
        if "proba_scale" in e:
            md.append(f"- **decode:** {e['decode']} (scale {e['proba_scale']})")
        if e["bands"] <= 16:
            md.append(f"- bands: {', '.join(e['band_descriptions'])}")
        md.append("")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(md))

    print(f"wrote {args.out}.json + {args.out}.md ({len(entries)} outputs)",
          flush=True)


if __name__ == "__main__":
    main()
