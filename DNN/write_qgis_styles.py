"""Write QGIS .qml sidecars so the output rasters open with the EE palette.

QGIS auto-loads `<raster>.qml` from beside `<raster>.tif` with no user action, so
dropping these next to the outputs is all it takes for `classified_2024.tif` to
open with the same colours as DNN/gee_display_inference.js instead of a grey
stretch. A `.clr` is written alongside for tools that want a plain colour table
(GDAL, ArcGIS, `gdaldem color-relief`).

The palette below is a transcription of `CLASSES` / `SETSIZE_RAMP_PALETTE` /
the pcal ramp in gee_display_inference.js. It is duplicated rather than parsed
out of the JS because a regex over someone else's source is a worse dependency
than a copy with a pointer to the original — but it means the two must be edited
together, hence the check in `_verify_against_js`.

Label space: raw codes are non-contiguous (8 -> 10) and a merged run drops 11
entirely, so entries are filtered to the classes the model actually emits
(`--model`, same contract as write_manifest.py). A palette listing a class the
raster cannot contain produces a legend entry that never appears on the map.

Run:
  PY DNN/write_qgis_styles.py --dir /data/P-Prosjekter2/154001_nyvest \
     --model models/dnn_final_moe8_merged.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parents[1]
JS = Path(__file__).resolve().parent / "gee_display_inference.js"

# ---- transcribed from gee_display_inference.js --------------------------------
# (code, EE colour, EE name). Names here are the EE script's own; the merged
# entry is relabelled in `resolve_classes` because code 2 changes meaning.
CLASSES = [
    (2,  "bdb76b", "bare"),
    (3,  "e8d63a", "cropland"),
    (4,  "1a7d34", "forest"),
    (5,  "a3d977", "grassland"),
    (6,  "c49a52", "scrub"),
    (7,  "5fbcd3", "wetland"),
    (8,  "2b5dbd", "water"),
    (10, "d93030", "settlement"),
    (11, "cde6a5", "sparse veg"),
    (12, "ffffff", "snow/ice"),
]
SETSIZE_RAMP = ["1a9850", "66bd63", "a6d96a", "d9ef8b", "ffffbf",
                "fee08b", "fdae61", "f46d43", "d73027", "7f0000"]
SETSIZE_WORST = "000000"          # set size 0 = empty/invalid conformal set
PCAL_RAMP = ["440154", "31688e", "35b779", "fde725"]   # viridis, 0..1
PCAL_SCALE = 60000

# When 11 is merged into 2 the class stops being "bare" in any useful sense —
# 91% of it is alpine sparse vegetation — so it gets its own label and a colour
# blended toward sparse-veg, otherwise a reader sees bare-ground khaki over the
# whole high plateau.
MERGED = {(11, 2): ("c8c98a", "bare ground + sparse veg")}


def _verify_against_js():
    """Fail loudly if the JS palette has drifted from the copy above."""
    if not JS.exists():
        return ["gee_display_inference.js not found — palette not cross-checked"]
    src = JS.read_text()
    warn = []
    for code, colour, name in CLASSES:
        pat = re.compile(r"\{code:\s*%d\s*,\s*name:\s*'([^']*)'\s*,\s*color:\s*'([0-9a-fA-F]{6})'"
                         % code)
        m = pat.search(src)
        if not m:
            warn.append(f"code {code}: not found in the JS CLASSES list")
        elif m.group(2).lower() != colour:
            warn.append(f"code {code} ({name}): JS says #{m.group(2)}, this file says #{colour}")
    for c in SETSIZE_RAMP + [SETSIZE_WORST]:
        if c not in src:
            warn.append(f"set-size colour #{c} is not in the JS ramp")
    return warn


def resolve_classes(model_path):
    """[(code, colour, label)] for the label space this model emits."""
    entries = list(CLASSES)
    if not model_path:
        return entries
    import torch
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    present = list(ck["classes"])
    out = []
    for code, colour, name in entries:
        if code not in present:
            continue
        for (src, dst), (mc, ml) in MERGED.items():
            if code == dst and dst in present and src not in present:
                colour, name = mc, ml
        out.append((code, colour, name))
    return out


# ------------------------------------------------------------------ QML writers
_HEAD = ('<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
         '<qgis version="3.34.0" styleCategories="AllStyleCategories">\n'
         '  <pipe>\n')
_TAIL = ('    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
         '    <huesaturation saturation="0" grayscaleMode="0" colorizeOn="0"/>\n'
         '    <rasterresampler maxOversampling="2"/>\n'
         '  </pipe>\n'
         '  <blendMode>0</blendMode>\n'
         '</qgis>\n')


def qml_paletted(entries, band=1):
    x = [_HEAD,
         f'    <rasterrenderer type="paletted" band="{band}" opacity="1" '
         f'alphaBand="-1" nodataColor="">\n',
         '      <rasterTransparency/>\n',
         '      <colorPalette>\n']
    for code, colour, label in entries:
        x.append(f'        <paletteEntry value="{code}" color="#{colour}" '
                 f'alpha="255" label="{escape(f"{code} {label}")}"/>\n')
    x += ['      </colorPalette>\n', '    </rasterrenderer>\n', _TAIL]
    return "".join(x)


def qml_exact(items, band=1, lo=None, hi=None):
    """Discrete colour map — one colour per integer value (EXACT shader)."""
    vals = [v for v, _, _ in items]
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    x = [_HEAD,
         f'    <rasterrenderer type="singlebandpseudocolor" band="{band}" '
         f'opacity="1" alphaBand="-1" classificationMin="{lo}" '
         f'classificationMax="{hi}" nodataColor="">\n',
         '      <rastershader>\n',
         '        <colorrampshader colorRampType="EXACT" classificationMode="2" '
         'clip="0" minimumValue="%s" maximumValue="%s">\n' % (lo, hi)]
    for v, colour, label in items:
        x.append(f'          <item value="{v}" color="#{colour}" alpha="255" '
                 f'label="{escape(label)}"/>\n')
    x += ['        </colorrampshader>\n', '      </rastershader>\n',
          '    </rasterrenderer>\n', _TAIL]
    return "".join(x)


def qml_interpolated(items, band=1, lo=0, hi=1):
    x = [_HEAD,
         f'    <rasterrenderer type="singlebandpseudocolor" band="{band}" '
         f'opacity="1" alphaBand="-1" classificationMin="{lo}" '
         f'classificationMax="{hi}" nodataColor="">\n',
         '      <rastershader>\n',
         '        <colorrampshader colorRampType="INTERPOLATED" '
         'classificationMode="1" clip="0" minimumValue="%s" maximumValue="%s">\n'
         % (lo, hi)]
    for v, colour, label in items:
        x.append(f'          <item value="{v}" color="#{colour}" alpha="255" '
                 f'label="{escape(label)}"/>\n')
    x += ['        </colorrampshader>\n', '      </rastershader>\n',
          '    </rasterrenderer>\n', _TAIL]
    return "".join(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directory holding the output rasters")
    ap.add_argument("--model", default=None,
                    help="the .pt used for inference; filters/renames the legend "
                         "to the classes it actually emits")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing .qml (they are hand-editable in QGIS, "
                         "so an existing one may be someone's customisation)")
    args = ap.parse_args()

    for w in _verify_against_js():
        print(f"  WARNING: {w}", flush=True)

    entries = resolve_classes(args.model)
    nclass = len(entries)
    print(f"palette: {nclass} classes {[c for c, _, _ in entries]}", flush=True)

    # set size: 1..nclass on the ramp, 0 = empty set -> black (as the EE script
    # does by shunting 0 into its own worst-case bin rather than the low end)
    step = max(1, len(SETSIZE_RAMP) // max(nclass, 1))
    ss_items = [(0, SETSIZE_WORST, "0  empty set (worst)")]
    for i in range(1, nclass + 1):
        colour = SETSIZE_RAMP[min(len(SETSIZE_RAMP) - 1, (i - 1) * step)]
        tag = "  singleton (best)" if i == 1 else ("  all classes" if i == nclass else "")
        ss_items.append((i, colour, f"{i}{tag}"))

    pcal_items = [(int(round(f * PCAL_SCALE)), c, f"{f:.2f}")
                  for f, c in zip((0.0, 1 / 3, 2 / 3, 1.0), PCAL_RAMP)]
    inset_items = [(0, "252525", "0  not in set"), (1, "1a9850", "1  in set")]
    # Change layer. Deliberately NOT a red/green pair: at this AOI's measured
    # error floor most of the "change" is classifier variance, and a saturated
    # red reads as a finding. No-change is muted, change is amber — visible,
    # but not asserting more than the data supports. See change_*.json.
    change_items = [(1, "eeeeee", "1  no change"),
                    (2, "e6a020", "2  change (see error floor in change_*.json)")]
    # Nature loss is graded by confidence, so the ramp encodes the tier rather
    # than the class: candidates that fail the screens stay pale because at this
    # AOI's error floor the raw candidate layer is ~1:1 with its own reverse-flow
    # control, and only tier 4 is worth putting in front of anyone.
    natloss_items = [(1, "f2f2f2", "1  no nature loss"),
                     (2, "f6dcc8", "2  candidate — fails the screens (likely noise)"),
                     (3, "e08040", "3  passes the screens, under MMU"),
                     (4, "a01010", "4  passes screens + MMU  <- the layer to use")]

    written = 0
    for tif in sorted(glob.glob(os.path.join(args.dir, "*.tif"))):
        stem = Path(tif).stem
        if stem.startswith("nature_loss_"):
            body, kind = (qml_exact(natloss_items, lo=1, hi=4),
                          "nature loss tier (band 1; see nature_loss_*.json)")
        elif stem.startswith("change_"):
            body, kind = qml_exact(change_items, lo=1, hi=2), "change / no change"
        elif "classified" in stem:
            body, kind = qml_paletted(entries), "paletted class map"
        elif stem.endswith("_setsize"):
            body, kind = qml_exact(ss_items, lo=0, hi=nclass), "conformal set size"
        elif stem.endswith("_pcal"):
            body, kind = (qml_interpolated(pcal_items, lo=0, hi=PCAL_SCALE),
                          f"calibrated proba (band 1; /{PCAL_SCALE})")
        elif stem.endswith("_inset"):
            body, kind = qml_exact(inset_items, lo=0, hi=1), "conformal membership"
        else:
            continue
        qml = str(Path(tif).with_suffix(".qml"))
        if os.path.exists(qml) and not args.force:
            print(f"  skip (exists): {os.path.basename(qml)}  — use --force")
            continue
        Path(qml).write_text(body)
        written += 1
        print(f"  {os.path.basename(qml):<34} {kind}")

    # portable colour table for the class map (GDAL / ArcGIS / gdaldem)
    clr = os.path.join(args.dir, "class_palette.clr")
    with open(clr, "w") as f:
        f.write("# code R G B  — nyvest land cover, palette from "
                "DNN/gee_display_inference.js\n")
        for code, colour, label in entries:
            r, g, b = (int(colour[i:i + 2], 16) for i in (0, 2, 4))
            f.write(f"{code} {r} {g} {b}  # {label}\n")
        f.write("0 0 0 0  # nodata\n")
    print(f"  {os.path.basename(clr):<34} portable colour table")
    print(f"\n{written} .qml written -> QGIS applies them automatically when the "
          f"raster is added.", flush=True)


if __name__ == "__main__":
    main()
