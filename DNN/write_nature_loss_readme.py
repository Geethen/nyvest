"""Write NATURE_LOSS_README.md next to the nature-loss rasters.

The numbers, class sets, screen settings and file names are read back out of
the `nature_loss*.json` reports and the GeoTIFF headers rather than typed here,
for the same reason write_manifest.py works that way: a README that is edited by
hand drifts away from the rasters it describes on the first re-run, and a stale
area figure in a nature-accounting deliverable is worse than no figure. Only the
prose — what each screen means, and what the reverse-flow control is for — is
static, because that does not change when the layer is rebuilt.

Run it after make_nature_loss.py, once per output directory:
  PY DNN/write_nature_loss_readme.py --dir <folder with nature_loss*.json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date
from pathlib import Path

import rasterio

from make_nature_loss import CLASS_NAMES, SCREENS

# What each screen actually tests, in the reader's terms. Static prose: the
# mechanism does not change between runs, only which screens were selected.
SCREEN_DOC = {
    "SINGLE": ("both years produced a **singleton conformal prediction set** — "
               "the calibrated model could rule out every other class, twice"),
    "CFWD": ("the new class was **not in the first year's** conformal prediction "
             "set — the model did not merely prefer the old class in the old "
             "year, it excluded the new one"),
    "CREV": ("the old class is **not in the second year's** conformal set — the "
             "mirror of CFWD. Measured and rejected: on this AOI it is "
             "*anti*-informative, see the table below"),
    "PROB": ("the calibrated probability of the new class is high in the second "
             "year and low in the first — a margin test rather than a "
             "set-membership one"),
    "SPATIAL": ("at least N of the 8 neighbouring pixels changed the same way — "
                "isolated single pixels are what independent per-pixel model "
                "variance looks like"),
    "PLAUS": ("transitions that are artefacts by construction are dropped: water "
              "and snow/ice into cropland or settlement (shoreline, tide and "
              "snow date, not development), and grassland into cropland unless "
              "explicitly requested"),
}

INTRO = """\
# Nature loss, {years}

Rasters showing land that was **natural** in {y0} and **anthropogenic** in {y1},
derived from `classified_{y0}.tif` and `classified_{y1}.tif` and their conformal
uncertainty layers.

> **Read this before using the numbers.** A raw class-flip between two
> independently classified years is dominated by classifier variance, not by
> change on the ground. On this AOI the unscreened nature-loss layer has *more*
> reverse flow — settlement reverting to nature, which does not happen at scale
> in six years — than forward flow. Everything useful in these files comes from
> the screening described below, and even the screened layer carries a
> measurable false-positive rate. Use band 1 tier 4, and quote the residual.
"""

METHOD = """\
## What was done

1. **Pixel-wise comparison.** The two class maps share a grid pixel for pixel, so
   no resampling was involved. A pixel is a nature-loss *candidate* where the
   {y0} class is in the natural set and the {y1} class is in the anthropogenic
   set (both listed per layer above).

2. **The reverse flow is measured as a control.** Every candidate has a mirror:
   anthropogenic in {y0}, natural in {y1}. Settlement does not revert to forest
   over six years, so those pixels are almost entirely error, and under any given
   screen they estimate that screen's residual error in the forward direction.
   This is the `loss:rev` column throughout. A ratio of 1.0 means the layer is
   indistinguishable from noise.

3. **Per-pixel screens.** Each candidate was tested by every screen below, and
   the results stored as a bitmask in band 3 — so a different combination can be
   costed later from the stored raster without re-running the sweep.

{screen_list}

4. **Minimum mapping unit.** The selected screen combination was then labelled
   into connected components (8-connectivity, stitched across strip boundaries),
   and patches below the MMU dropped. A minimum mapping unit is a property of a
   patch, not of a pixel, so this happens after the per-pixel screens — and it is
   the single biggest honest improvement in the whole chain. It is applied to the
   control direction too, so the two stay comparable.

5. **Tiers written, nothing deleted.** Failing candidates are kept as tier 2
   rather than erased, so the layer can be re-thresholded and the discarded
   material inspected.
"""

USAGE = """\
## Using the layers

* **Drop the `.tif` into QGIS.** The `.qml` beside it loads automatically and
  colours band 1 by tier — tier 4 dark red, tier 3 orange, tier 2 pale.
* **Quote tier 4**, and quote the residual error with it (the control column in
  the results table). Tier 4 is a gross figure, not a net one.
* **Band 2 gives the breakdown** — which natural class was lost, per pixel.
* **To re-threshold without re-running:** band 3 holds the screen bitmask. A
  pixel passed screen X where `band3 & bit(X) == bit(X)`. Bit 6 (value 64) is
  not a screen — it marks the reverse-direction control pixels.
* **Do not** difference the two years yourself from `classified_*.tif` and treat
  the result as change. That is `change_{y0}_{y1}.tif`, and its own report
  records why it is not a change product.

## Known limitations

* The `loss:rev` ratio is a self-consistency argument, not an accuracy
  assessment. An unbiased area with a confidence interval needs a stratified
  sample over the tier bands with reference interpretation (Olofsson et al.).
* PROB and CFWD test confidence in the *destination* class, and settlement and
  cropland are much easier classes for this model than mire or sparse veg, so
  both screens flatter themselves in the forward direction. SINGLE, SPATIAL and
  PLAUS treat the two directions alike, and their ratios are the honest ones.
* Both years were classified by the same model from the same lidar epoch. That
  is a shared confound between the two maps, not an independent observation of
  each year.
* Grassland/cropland is a definitional boundary, not a physical one — rotation,
  fallow and mowing state move a field across it with no land converted.
"""


def _fmt_classes(codes, names=None):
    names = names or {}
    return ", ".join(f"`{c}` {names.get(str(c), names.get(c, CLASS_NAMES.get(int(c), '?')))}"
                     for c in codes)


def _table(rows, headers, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _cmd(rep, path, d):
    """Reconstruct the command that produced this layer.

    Spelled with the real output directory rather than `.`: the script path is
    relative to the repo root and the data lives elsewhere, so a `--dir .` here
    would be a command that cannot run from either place.
    """
    stem = Path(path).stem                      # nature_loss[_tag]_y0_y1
    y0, y1 = rep["years"]
    tag = stem[len("nature_loss"):-len(f"_{y0}_{y1}")].strip("_")
    a = [f"PY DNN/make_nature_loss.py --dir {d}",
         f"--years {y0} {y1}",
         "--model models/dnn_final_moe8_merged.pt",
         "--natural " + ",".join(str(c) for c in rep["natural"]),
         "--anthro " + ",".join(str(c) for c in rep["anthropogenic"]),
         "--screen " + (",".join(rep["selected_screen"]) or "NONE"),
         f"--nbr {rep['nbr']}", f"--mmu-px {rep['mmu_px']}"]
    if tag:
        a.append(f"--tag {tag}")
    if rep.get("grass_to_crop_included"):
        a.append("--include-grass-to-crop")
    return " \\\n     ".join([a[0], " ".join(a[1:3])] + a[3:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="NATURE_LOSS_README.md")
    args = ap.parse_args()

    d = Path(args.dir)
    reports = []
    for j in sorted(glob.glob(str(d / "nature_loss*.json"))):
        rep = json.loads(Path(j).read_text())
        if rep.get("kind") != "nature_loss":
            continue
        tif = Path(rep["raster"])
        tif = tif if tif.exists() else d / (Path(j).stem + ".tif")
        if not tif.exists():
            print(f"  ! {Path(j).name} describes a raster that is not here — skipped")
            continue
        reports.append((Path(j), tif, rep))
    if not reports:
        raise SystemExit(f"no nature_loss*.json in {d}")

    y0, y1 = reports[0][2]["years"]
    L = [INTRO.format(years=f"{y0}–{y1}", y0=y0, y1=y1)]

    # ---- files ---------------------------------------------------------------
    L.append("## Files\n")
    rows = []
    for j, tif, rep in reports:
        with rasterio.open(tif) as s:
            geom = (f"{s.width} × {s.height}, {s.res[0]:g} m, "
                    f"{s.crs.to_string()}, {s.dtypes[0]}, nodata {int(s.nodata)}")
        qml = tif.with_suffix(".qml")
        rows.append([f"`{tif.name}`", f"{tif.stat().st_size/1e6:.0f} MB", geom])
        rows.append([f"`{j.name}`", f"{j.stat().st_size/1e3:.0f} kB",
                     "full screen comparison and per-class areas for the above"])
        if qml.exists():
            rows.append([f"`{qml.name}`", f"{qml.stat().st_size/1e3:.0f} kB",
                         "QGIS style for band 1; loads automatically"])
    L.append(_table(rows, ["file", "size", "what it is"]) + "\n")

    # ---- bands ---------------------------------------------------------------
    with rasterio.open(reports[0][1]) as s:
        descs = s.descriptions
    L.append("## Bands\n")
    L.append(_table([
        ["1", f"`{descs[0]}`",
         "**0** nodata (either year unclassified) &middot; **1** valid, no nature "
         "loss &middot; **2** candidate that fails the screens (mostly classifier "
         "noise — not a finding) &middot; **3** passes the screens but below "
         "the MMU &middot; **4** passes screens **and** MMU &mdash; **this is the "
         "layer to use**"],
        ["2", f"`{descs[1]}`",
         "the natural class that was lost, as a raw grunnkart code. Only codes "
         "from that layer's own natural set can appear (listed per layer below); "
         "across the layers here that is "
         + _fmt_classes(sorted({c for _, _, r in reports for c in r["natural"]}))
         + ". `0` where band 1 is not a loss candidate"],
        ["3", f"`{descs[2]}`",
         "bitmask of which screens the pixel passed — "
         + ", ".join(f"`{v}` {k}" for k, v in reports[0][2]["screen_bits"].items())
         + ". Bit 6 (`64`) is **not** a screen: it marks the reverse-direction "
           "control pixels used to estimate the false-positive rate"],
    ], ["band", "name", "values"]) + "\n")

    # ---- per-layer definition + results --------------------------------------
    for j, tif, rep in reports:
        mmu, sc = rep["mmu"], rep["selected_screen"]
        L.append(f"## `{tif.name}`\n")
        L.append(f"**Natural** — {_fmt_classes(rep['natural'])}  \n"
                 f"**Anthropogenic** — {_fmt_classes(rep['anthropogenic'].keys(), rep['anthropogenic'])}  \n"
                 f"**Grassland → cropland** — "
                 f"{'included' if rep['grass_to_crop_included'] else 'excluded'}  \n"
                 f"**Screens applied** — {'+'.join(sc) if sc else 'none'}, "
                 f"then MMU ≥ {rep['mmu_px']} px "
                 f"({rep['mmu_px']*rep['ha_per_px']:.2f} ha)\n")

        raw = next(r for r in rep["screen_comparison"] if r["mask"] == 0)
        sel_mask = sum(rep["screen_bits"][n] for n in sc)
        sel = next((r for r in rep["screen_comparison"] if r["mask"] == sel_mask), None)
        L.append("### Result\n")
        L.append(_table([
            ["raw candidates, no screening", f"{raw['loss_ha']:,.0f}",
             f"{raw['gain_ha']:,.0f}", f"**{raw['loss_per_reverse']}**", ""],
            [f"tier 3 — {'+'.join(sc) if sc else 'none'}",
             f"{sum(rep['to_class_screened'].values())*rep['ha_per_px']:,.0f}",
             f"{sel['gain_ha']:,.0f}" if sel else "",
             ("∞" if sel["loss_per_reverse"] is None
              else f"{sel['loss_per_reverse']:.2f}") if sel else "", ""],
            ["**tier 4 — screens + MMU**",
             f"**{mmu['loss_kept_px']*rep['ha_per_px']:,.0f}**",
             f"{mmu['reverse_kept_px']*rep['ha_per_px']:,.0f}",
             f"**{mmu['loss_per_reverse']}**", f"{mmu['loss_patches']:,}"],
        ], ["layer", "loss (ha)", "reverse control (ha)", "loss:rev", "patches"],
            [":---", "---:", "---:", "---:", "---:"]))
        L.append(f"\nRead tier 4 as **~{mmu['loss_kept_px']*rep['ha_per_px']:,.0f} ha "
                 f"gross with an estimated ~{mmu['reverse_kept_px']*rep['ha_per_px']:,.0f} ha "
                 f"still false**, i.e. roughly "
                 f"{(mmu['loss_kept_px']-mmu['reverse_kept_px'])*rep['ha_per_px']:,.0f} ha "
                 f"net. The MMU step discarded "
                 f"{mmu['loss_dropped_px']*rep['ha_per_px']:,.0f} ha spread over "
                 f"{mmu['loss_components']-mmu['loss_patches']:,} sub-MMU specks.\n")

        L.append("### Where the loss is (tier 3, before MMU)\n")
        rows = []
        for c, px in sorted(rep["from_class_screened"].items(),
                            key=lambda kv: -kv[1]):
            rows.append([f"`{c}` {CLASS_NAMES.get(int(c), '?')}",
                         f"{rep['from_class_raw'][c]*rep['ha_per_px']:,.0f}",
                         f"{px*rep['ha_per_px']:,.0f}"])
        L.append(_table(rows, ["lost from", "raw (ha)", "screened (ha)"],
                        [":---", "---:", "---:"]) + "\n")

        L.append("### Every screen, costed against the control\n")
        rows = []
        for r in rep["screen_comparison"]:
            rows.append([r["screen"], f"{r['loss_ha']:,.0f}", f"{r['gain_ha']:,.0f}",
                         "∞" if r["loss_per_reverse"] is None
                         else f"{r['loss_per_reverse']:.2f}",
                         f"{r['kept_pct']:.1f}%"])
        L.append(_table(rows, ["screen", "loss (ha)", "reverse (ha)", "loss:rev",
                               "kept"], [":---", "---:", "---:", "---:", "---:"]))
        L.append(f"\nRebuilt with, from the repo root (`PY` = the venv "
                 f"interpreter, see CLAUDE.md):\n\n```bash\n{_cmd(rep, tif, d)}"
                 f"\n```\n")

    # ---- method + usage ------------------------------------------------------
    used = sorted({s for _, _, r in reports for s in r["selected_screen"]})
    lines = []
    for s in SCREENS:
        mark = " *(selected)*" if s in used else ""
        lines.append(f"   * **{s}**{mark} — {SCREEN_DOC[s]}.")
    L.append(METHOD.format(y0=y0, y1=y1, screen_list="\n".join(lines)))
    L.append(USAGE.format(y0=y0, y1=y1))
    L.append(f"\n---\n\nGenerated by `DNN/write_nature_loss_readme.py` on "
             f"{date.today().isoformat()} from the `nature_loss*.json` reports "
             f"beside these rasters. Do not hand-edit — re-run it instead.\n")

    out = d / args.out
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({out.stat().st_size/1e3:.1f} kB, "
          f"{len(reports)} layer(s): "
          f"{', '.join(t.name for _, t, _ in reports)})")


if __name__ == "__main__":
    main()
