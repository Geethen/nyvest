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

# Raw class codes the model emits -> human labels (from DNN/confusion_matrix.py).
# Codes 1 and 9 never appear: training merges 1->2 and 9->8 (see DNN/README.md).
CLASS_LABELS = {2: "bare", 3: "cropland", 4: "forest", 5: "grassland",
                6: "scrub", 7: "wetland", 8: "water", 10: "settle",
                11: "infra", 12: "snow/ice"}

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
}


def _role(stem: str) -> str:
    for key, note in ROLES.items():
        if key.startswith("_"):
            if stem.endswith(key):
                return note
        elif stem.startswith(key) or key in stem:
            return note
    return "(undocumented output — inspect band descriptions)"


def describe(path: str) -> dict:
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
        entry["class_legend"] = {str(k): v for k, v in CLASS_LABELS.items()}
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory of output rasters")
    ap.add_argument("--out", required=True,
                    help="manifest path stem (writes <out>.json + <out>.md)")
    ap.add_argument("--glob", default="*.tif",
                    help="which files to describe (default *.tif; VRTs skipped "
                         "unless matched)")
    args = ap.parse_args()

    import glob as globmod
    paths = sorted(globmod.glob(os.path.join(args.dir, args.glob)))
    paths = [p for p in paths if not p.endswith(".aux.xml")]
    entries = []
    for p in paths:
        try:
            entries.append(describe(p))
        except rasterio.errors.RasterioIOError:
            continue

    manifest = {
        "generated": date.today().isoformat(),
        "dir": os.path.abspath(args.dir),
        "class_legend": {str(k): v for k, v in CLASS_LABELS.items()},
        "note": "Class codes 1 and 9 are absent by design: training merges "
                "1->2 and 9->8 (see DNN/README.md). Codes are RAW, not 0..N.",
        "outputs": entries,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(manifest, f, indent=2)

    # readable markdown
    md = [f"# Inference outputs — {manifest['dir']}",
          f"\nGenerated {manifest['generated']}.\n",
          "## Class legend (raw codes)\n",
          "| code | class |", "|---|---|"]
    for k, v in CLASS_LABELS.items():
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
