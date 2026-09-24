"""Nature-loss layer for the 2018/2024 pair, and the screens that decide how
much of it to believe.

`make_change_layer.py` answers "did the class flip?". That question is the wrong
one for a nature-accounting product: most flips in this AOI are within-nature
confusions (scrub<->bare, forest<->mire) that the classifier gets wrong twice as
often as the ground actually changes. This script asks the narrower question
instead — did a NATURAL class become an ANTHROPOGENIC one? — and then spends
most of its effort measuring its own false-positive rate.

Definition
    natural       2 bare+sparse, 4 forest, 5 grassland, 6 scrub, 7 wetland,
                  8 water, 12 snow/ice
    anthropogenic 3 cropland, 10 settlement
    nature loss   natural(y0) -> anthropogenic(y1)

  Both sets are `--natural` / `--anthro`, because what counts as nature is a
  policy question rather than a property of the raster. A useful second run is
  the restrictive one, `--natural 2,4,5,6 --anthro 10 --tag strict`: it drops
  mire, open water and permanent snow from the numerator and cropland from the
  denominator, leaving only land that was vegetated or bare and is now built.
  That layer is a third the size and roughly twice as clean, because the three
  classes it drops are the ones whose boundaries move with tide, water level
  and snow date rather than with development.

  Grassland -> cropland is EXCLUDED by default (`--include-grass-to-crop` to
  add it). 5<->3 is the largest flip in the whole matrix and runs 1.9M px one
  way against 1.2M the other; rotation, fallow and mowing state move a field
  across that boundary without any land being converted, so counting it as
  nature loss would swamp the layer with the one transition we can least
  defend. Grassland -> settlement is kept: building on pasture is real loss.

The false-positive control
    The reverse flow — anthropogenic(y0) -> natural(y1) — is the key number.
    Settlement does not revert to forest over six years, so under any screen the
    surviving reverse pixels are almost entirely error, and they estimate the
    error in the forward direction under that same screen. A screen is only
    worth using if it removes reverse flow faster than it removes loss. The
    unscreened layer fails this test outright: this AOI's built area SHRINKS
    between the two epochs.

Screens (independent, recorded as a bitmask in band 3 so any combination can
be costed after the fact from one pass):

    SINGLE  both years produced a singleton conformal set — confident twice
    CFWD    the new class was NOT in year0's conformal set (the change is
            conformally significant going forward)
    CREV    the old class is NOT in year1's conformal set (significant back)
    PROB    p_y1(new) >= --p-hi and p_y0(new) <= --p-lo on the calibrated
            probabilities — a margin test rather than a set-membership one
    SPATIAL >= --nbr of the 8 neighbours are candidates in the same direction;
            kills isolated speckle, which is what independent per-pixel model
            variance mostly looks like
    PLAUS   drops transitions that are artefacts by construction: water and
            snow/ice into cropland or settlement (shoreline, tide and seasonal
            snow), and grass->crop when it is not requested

  MMU is applied after the pass as connected-component labelling on the chosen
  screen combination (`--mmu-px`, default 10 px = 0.1 ha), because a minimum
  mapping unit is a property of a patch, not of a pixel. It is costed on the
  reverse direction too, which is why band 3 keeps bit 6 for control pixels.

Read the loss:reverse ratio with one asymmetry in mind: PROB and CFWD test
confidence in the DESTINATION class, and settlement and cropland are much
easier classes for this model than mire or sparse veg, so both screens flatter
themselves in the forward direction. SINGLE, SPATIAL and PLAUS treat the two
directions alike, and their ratios are the ones to trust.

Output `nature_loss[_<tag>]_<y0>_<y1>.tif`, 3 bands:
    1 loss_tier  0 nodata, 1 no loss, 2 candidate (fails --screen),
                 3 passes --screen but under MMU, 4 passes both  <- use this one
    2 from_class the natural class that was lost (0 elsewhere)
    3 screens    the screen bitmask, so a different combination can be
                 re-thresholded without re-running the sweep

Run:
  PY DNN/make_nature_loss.py --dir <folder with classified_*.tif> \
     --years 2018 2024 --model models/dnn_final_moe8_merged.pt
  PY DNN/make_nature_loss.py --dir <same> --model <same> \
     --natural 2,4,5,6 --anthro 10 --tag strict
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
from scipy import ndimage                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Defaults. Both sets are `--natural` / `--anthro` on the command line, because
# what counts as "nature" is a policy question, not a property of the raster:
# a nature-accounting audience that will not defend mire, open water and
# permanent snow as convertible land wants (2,4,5,6) -> (10) instead, and gets
# a different, smaller, better-behaved layer for it.
NATURAL = (2, 4, 5, 6, 7, 8, 12)
ANTHRO = {3: "cropland", 10: "settlement"}
CLASS_NAMES = {2: "bare+sparse", 3: "cropland", 4: "forest", 5: "grassland",
               6: "scrub", 7: "wetland", 8: "water", 10: "settlement",
               12: "snow/ice"}

NODATA, NO_LOSS, CAND, SCREENED, HIGH = 0, 1, 2, 3, 4

# bit -> name. Order matters only for the report's cumulative column.
SCREENS = ["SINGLE", "CFWD", "CREV", "PROB", "SPATIAL", "PLAUS"]
BIT = {n: 1 << i for i, n in enumerate(SCREENS)}
# bit 6 is not a screen: it flags the reverse-direction control pixels so the
# MMU pass can cost them on the same footing as the forward ones.
REVERSE_BIT = 1 << 6

PCAL_SCALE = 60000          # write_qgis_styles.PCAL_SCALE / predict_raster
HA_PER_PX = 0.01            # 10 m pixels


def _band_index(descriptions, cls):
    """1-based band holding class `cls` in the uq_* rasters ('inset_c10')."""
    for i, d in enumerate(descriptions):
        if d and d.rsplit("_c", 1)[-1] == str(cls):
            return i + 1
    raise SystemExit(f"no band for class {cls} in {descriptions}")


def _nbr_count(mask):
    """8-neighbour count of a bool array, zero-padded at the array edge."""
    p = np.pad(mask.astype(np.uint8), 1)
    out = np.zeros(mask.shape, np.uint8)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            if dr == 1 and dc == 1:
                continue
            out += p[dr:dr + mask.shape[0], dc:dc + mask.shape[1]]
    return out


def _combo_count(hist, required):
    """Pixels whose bitmask is a superset of `required`, from a 64-bin hist."""
    idx = np.arange(len(hist))
    return int(hist[(idx & required) == required].sum())


# ----------------------------------------------------------------- main sweep
def sweep(args, d, y0, y1):
    p0, p1 = d / f"classified_{y0}.tif", d / f"classified_{y1}.tif"
    s0, s1 = d / f"uq_{y0}_setsize.tif", d / f"uq_{y1}_setsize.tif"
    i0, i1 = d / f"uq_{y0}_inset.tif", d / f"uq_{y1}_inset.tif"
    c0, c1 = d / f"uq_{y0}_pcal.tif", d / f"uq_{y1}_pcal.tif"
    out = d / f"nature_loss{args.tag_part}_{y0}_{y1}.tif"

    have_ss = s0.exists() and s1.exists()
    have_in = i0.exists() and i1.exists()
    have_pc = c0.exists() and c1.exists() and not args.no_prob
    if not have_ss:
        print("  ! no uq_*_setsize.tif — SINGLE screen unavailable")
    if not have_in:
        print("  ! no uq_*_inset.tif — CFWD/CREV screens unavailable")
    if not have_pc:
        print("  ! pcal skipped — PROB screen unavailable")

    natural, anthro = args.natural, args.anthro
    grass_crop = args.include_grass_to_crop
    # (from, to) pairs that are artefacts by construction; see PLAUS above.
    # Pairs outside the chosen sets simply never fire, so the rule needs no
    # per-definition editing.
    implausible = {(8, 3), (8, 10), (12, 3), (12, 10)}
    if not grass_crop and 5 in natural and 3 in anthro:
        implausible.add((5, 3))

    nat = np.array(natural, dtype=np.int16)
    ant = np.array(sorted(anthro), dtype=np.int16)

    hi = int(round(args.p_hi * PCAL_SCALE))
    lo = int(round(args.p_lo * PCAL_SCALE))

    t0 = time.perf_counter()
    with rasterio.open(p0) as a, rasterio.open(p1) as b:
        if (a.width, a.height, a.crs, a.transform) != (b.width, b.height, b.crs, b.transform):
            raise SystemExit(f"{p0.name} and {p1.name} are not on the same grid.")
        W, H = a.width, a.height
        r_beg = args.test_start
        H_scan = min(H, r_beg + args.test_rows) if args.test_rows else H
        prof = a.profile.copy()
        prof.update(driver="GTiff", count=3, dtype="uint8", nodata=NODATA,
                    compress="deflate", predictor=2, tiled=True,
                    blockxsize=512, blockysize=512, bigtiff="IF_SAFER")

        hist_loss = np.zeros(64, np.int64)      # bitmask histogram, forward
        hist_gain = np.zeros(64, np.int64)      # ... and the reverse control
        # per (from_class, screened?) area, forward direction only
        from_raw = {c: 0 for c in natural}
        from_scr = {c: 0 for c in natural}
        to_raw = {c: 0 for c in anthro}
        to_scr = {c: 0 for c in anthro}
        n_valid = 0

        opened = []
        try:
            ss = [rasterio.open(s0), rasterio.open(s1)] if have_ss else None
            ins = [rasterio.open(i0), rasterio.open(i1)] if have_in else None
            pcs = [rasterio.open(c0), rasterio.open(c1)] if have_pc else None
            opened = [x for grp in (ss, ins, pcs) if grp for x in grp]

            if have_in:
                ib = {c: _band_index(ins[0].descriptions, c) for c in (*natural, *anthro)}
            if have_pc:
                pb = {c: _band_index(pcs[0].descriptions, c) for c in anthro}

            req = args.screen_mask
            with rasterio.open(out, "w", **prof) as dst:
                dst.set_band_description(1, "loss_tier")
                dst.set_band_description(2, "from_class")
                dst.set_band_description(3, "screens")
                for row in range(r_beg, H_scan, args.rows):
                    h = min(args.rows, H - row)
                    # one-row halo so SPATIAL sees across the strip seam
                    r0 = max(0, row - 1)
                    r1 = min(H, row + h + 1)
                    win = Window(0, r0, W, r1 - r0)
                    top = row - r0                     # rows of halo above
                    sl = slice(top, top + h)

                    A = a.read(1, window=win)
                    B = b.read(1, window=win)
                    valid = (A != 0) & (B != 0)

                    a_nat = np.isin(A, nat)
                    b_ant = np.isin(B, ant)
                    a_ant = np.isin(A, ant)
                    b_nat = np.isin(B, nat)
                    loss = valid & a_nat & b_ant
                    gain = valid & a_ant & b_nat
                    if not grass_crop and 5 in natural and 3 in anthro:
                        # keep the pair out of BOTH directions, else the control
                        # is measured on a transition the product does not claim
                        loss &= ~((A == 5) & (B == 3))
                        gain &= ~((A == 3) & (B == 5))
                    cand = loss | gain

                    m = np.zeros(A.shape, np.uint8)
                    if have_ss:
                        SA = ss[0].read(1, window=win)
                        SB = ss[1].read(1, window=win)
                        m |= np.where((SA == 1) & (SB == 1), BIT["SINGLE"], 0).astype(np.uint8)
                    if have_in:
                        I0 = ins[0].read(window=win)
                        I1 = ins[1].read(window=win)
                        # CFWD: new class not in y0's set. CREV: old not in y1's.
                        fwd = np.zeros(A.shape, bool)
                        rev = np.zeros(A.shape, bool)
                        for c in (*natural, *anthro):
                            sel = cand & (B == c)
                            if sel.any():
                                fwd |= sel & (I0[ib[c] - 1] == 0)
                            sel = cand & (A == c)
                            if sel.any():
                                rev |= sel & (I1[ib[c] - 1] == 0)
                        m |= (fwd * BIT["CFWD"]).astype(np.uint8)
                        m |= (rev * BIT["CREV"]).astype(np.uint8)
                        del I0, I1
                    if have_pc:
                        prob = np.zeros(A.shape, bool)
                        for c in anthro:
                            sel = cand & (B == c)
                            if sel.any():
                                P0 = pcs[0].read(pb[c], window=win)
                                P1 = pcs[1].read(pb[c], window=win)
                                prob |= sel & (P1 >= hi) & (P0 <= lo)
                                del P0, P1
                        m |= (prob * BIT["PROB"]).astype(np.uint8)
                    # SPATIAL: neighbours in the same direction only
                    sp = ((_nbr_count(loss) >= args.nbr) & loss) | \
                         ((_nbr_count(gain) >= args.nbr) & gain)
                    m |= (sp * BIT["SPATIAL"]).astype(np.uint8)
                    # PLAUS
                    pl = cand.copy()
                    for f, t in implausible:
                        pl &= ~(((A == f) & (B == t)) | ((A == t) & (B == f)))
                    m |= (pl * BIT["PLAUS"]).astype(np.uint8)

                    m[~cand] = 0
                    m |= (gain * REVERSE_BIT).astype(np.uint8)
                    # `sl` trims the halo: the strips overlap by a row so that
                    # SPATIAL can see across the seam, and counting the padded
                    # window would tally those rows in two strips.
                    hist_loss += np.bincount(m[sl][loss[sl]] & 63,
                                             minlength=64).astype(np.int64)
                    hist_gain += np.bincount(m[sl][gain[sl]] & 63,
                                             minlength=64).astype(np.int64)

                    passed = (m & req) == req
                    tier = np.where(valid, NO_LOSS, NODATA).astype(np.uint8)
                    tier[loss] = CAND
                    tier[loss & passed] = SCREENED
                    frm = np.where(loss, A, 0).astype(np.uint8)

                    n_valid += int(valid[sl].sum())
                    for c in natural:
                        sub = loss[sl] & (A[sl] == c)
                        from_raw[c] += int(sub.sum())
                        from_scr[c] += int((sub & passed[sl]).sum())
                    for c in anthro:
                        sub = loss[sl] & (B[sl] == c)
                        to_raw[c] += int(sub.sum())
                        to_scr[c] += int((sub & passed[sl]).sum())

                    wout = Window(0, row, W, h)
                    dst.write(tier[sl], 1, window=wout)
                    dst.write(frm[sl], 2, window=wout)
                    dst.write(m[sl], 3, window=wout)

                    if (row // args.rows) % 4 == 0:
                        el = time.perf_counter() - t0
                        print(f"  {row+h}/{H_scan} rows  {el:.0f}s  "
                              f"loss {hist_loss.sum()/1e6:.2f}M px  "
                              f"reverse {hist_gain.sum()/1e6:.2f}M px", flush=True)
        finally:
            for x in opened:
                x.close()

    return dict(out=out, W=W, H=H, hist_loss=hist_loss, hist_gain=hist_gain,
                from_raw=from_raw, from_scr=from_scr, to_raw=to_raw, to_scr=to_scr,
                n_valid=n_valid, have=dict(ss=have_ss, ins=have_in, pc=have_pc),
                secs=time.perf_counter() - t0)


# --------------------------------------------------------------------- MMU
def _components(src, sel_fn, min_px, rows=4096):
    """Connected components of a sparse boolean mask over a whole raster.

    Returns (keep_lut_per_strip, kept_px, dropped_px, n_kept_patches, n_total).
    The label array for 3.3 Gpx will not fit, so this labels strip by strip and
    stitches with a union-find over the seam rows — a per-strip label alone
    would cut every patch that straddles a seam.
    """
    parent, sizes, strips = {}, {}, []

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    off, prev = 0, None
    H, W = src.height, src.width
    for row in range(0, H, rows):
        h = min(rows, H - row)
        sel = sel_fn(src.read(window=Window(0, row, W, h)))
        lab, n = ndimage.label(sel)
        lab = lab.astype(np.int64)
        lab[lab > 0] += off
        if n:
            u, c = np.unique(lab[lab > 0], return_counts=True)
            for l, k in zip(u.tolist(), c.tolist()):
                sizes[l] = sizes.get(l, 0) + k
                parent.setdefault(l, l)
        if prev is not None and n:
            cur = lab[0]
            for sh in (0, 1, -1):                       # 8-connectivity over the seam
                p = prev if sh == 0 else np.roll(prev, sh)
                if sh == 1:
                    p = p.copy(); p[0] = 0
                elif sh == -1:
                    p = p.copy(); p[-1] = 0
                bb = (p > 0) & (cur > 0)
                if bb.any():
                    for x, y in set(zip(p[bb].tolist(), cur[bb].tolist())):
                        union(x, y)
        prev = lab[-1].copy()
        strips.append((row, h, off, n))
        off += n

    tot = {}
    for l, c in sizes.items():
        r = find(l)
        tot[r] = tot.get(r, 0) + c
    keep_roots = {r for r, c in tot.items() if c >= min_px}
    kept_px = sum(c for r, c in tot.items() if r in keep_roots)
    luts = {}
    for row, h, o, n in strips:
        lut = np.zeros(n + 1, bool)
        for l in range(1, n + 1):
            if find(o + l) in keep_roots:
                lut[l] = True
        luts[row] = lut
    return (luts, strips, kept_px, sum(sizes.values()) - kept_px,
            len(keep_roots), len(tot))


def apply_mmu(path, min_px, req, rows=4096):
    """Promote loss tier SCREENED -> HIGH for patches of >= min_px, and cost the
    same rule on the reverse-direction control so the two stay comparable."""
    def fwd(blk):
        return (blk[0] == SCREENED) & (((blk[2] & 63) & req) == req)

    def rev(blk):
        return ((blk[2] & REVERSE_BIT) > 0) & (((blk[2] & 63) & req) == req)

    with rasterio.open(path) as s:
        f = _components(s, fwd, min_px, rows)
        g = _components(s, rev, min_px, rows)

    luts, strips = f[0], f[1]
    with rasterio.open(path, "r+") as s:
        for row, h, o, n in strips:
            if not n:
                continue
            w = Window(0, row, s.width, h)
            blk = s.read(window=w)
            sel = fwd(blk)
            if sel.any():
                lab, _ = ndimage.label(sel)
                t = blk[0]
                t[luts[row][lab] & sel] = HIGH
                s.write(t, 1, window=w)

    return dict(min_px=min_px,
                loss_kept_px=f[2], loss_dropped_px=f[3],
                loss_patches=f[4], loss_components=f[5],
                reverse_kept_px=g[2], reverse_dropped_px=g[3],
                reverse_patches=g[4], reverse_components=g[5])


# ------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--years", nargs=2, type=int, default=[2018, 2024])
    ap.add_argument("--model", default=None, help="for class labels in the report")
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--screen", default="CFWD,SPATIAL,PLAUS",
                    help="comma-separated screens ALL must pass for tier 3 "
                         f"({'|'.join(SCREENS)}, or NONE)")
    ap.add_argument("--nbr", type=int, default=3, help="SPATIAL: min of 8 neighbours")
    ap.add_argument("--p-hi", type=float, default=0.60, help="PROB: p_y1(new) >=")
    ap.add_argument("--p-lo", type=float, default=0.20, help="PROB: p_y0(new) <=")
    ap.add_argument("--mmu-px", type=int, default=10, help="min patch size, px (10 = 0.1 ha)")
    ap.add_argument("--natural", default=",".join(str(c) for c in NATURAL),
                    help="class codes counted as nature (comma-separated)")
    ap.add_argument("--anthro", default=",".join(str(c) for c in sorted(ANTHRO)),
                    help="class codes counted as anthropogenic (comma-separated)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output name, so a second definition does "
                         "not overwrite the first (nature_loss_<tag>_<y0>_<y1>.tif)")
    ap.add_argument("--include-grass-to-crop", action="store_true")
    ap.add_argument("--no-prob", action="store_true", help="skip the pcal read (slowest)")
    ap.add_argument("--test-rows", type=int, default=0,
                    help="debug: stop after N rows (partial raster, partial stats)")
    ap.add_argument("--test-start", type=int, default=0, help="debug: first row")
    args = ap.parse_args()

    names = [s for s in args.screen.replace(" ", "").split(",") if s and s != "NONE"]
    bad = [s for s in names if s not in BIT]
    if bad:
        raise SystemExit(f"unknown screen(s) {bad}; pick from {SCREENS}")
    args.screen_mask = sum(BIT[s] for s in names)

    def _codes(spec, what):
        try:
            cs = tuple(int(x) for x in spec.replace(" ", "").split(",") if x)
        except ValueError:
            raise SystemExit(f"--{what} must be comma-separated integers, got {spec!r}")
        unknown = [c for c in cs if c not in CLASS_NAMES]
        if not cs or unknown:
            raise SystemExit(f"--{what}: {unknown or 'empty'} not in "
                             f"{sorted(CLASS_NAMES)}")
        return cs

    args.tag_part = ("_" + args.tag.strip("_")) if args.tag.strip("_") else ""
    args.natural = _codes(args.natural, "natural")
    args.anthro = {c: CLASS_NAMES[c] for c in _codes(args.anthro, "anthro")}
    both = set(args.natural) & set(args.anthro)
    if both:
        raise SystemExit(f"class(es) {sorted(both)} are in both --natural and "
                         f"--anthro; a class cannot be its own conversion target")

    d = Path(args.dir)
    y0, y1 = args.years

    labels = {}
    if args.model:
        try:
            from write_qgis_styles import resolve_classes
            labels = {c: n for c, _, n in resolve_classes(args.model)}
        except Exception as e:
            print(f"  (no labels: {e})")

    print(f"nature loss {y0} -> {y1}")
    print(f"  natural      {[f'{c} {labels.get(c, CLASS_NAMES[c])}' for c in args.natural]}")
    print(f"  anthropogenic{[f'{c} {n}' for c, n in args.anthro.items()]}")
    print(f"  grass->crop  {'INCLUDED' if args.include_grass_to_crop else 'excluded'}")
    print(f"  screen       {'+'.join(names) if names else 'NONE'}  mmu {args.mmu_px} px\n")

    r = sweep(args, d, y0, y1)
    hl, hg = r["hist_loss"], r["hist_gain"]
    tot_loss, tot_gain = int(hl.sum()), int(hg.sum())

    def ha(px):
        return px * HA_PER_PX

    print(f"\nswept {r['n_valid']:,} valid px in {r['secs']:.0f}s -> {r['out'].name}")
    print(f"\n{'='*94}\nSCREEN COMPARISON — the reverse flow is the false-positive control")
    print(f"{'='*94}")
    print(f"{'screen':<26}{'loss px':>12}{'loss ha':>11}{'reverse px':>12}"
          f"{'rev ha':>10}{'loss:rev':>10}{'kept':>8}")
    rows_rep = []

    def add(tag, mask):
        L = _combo_count(hl, mask)
        G = _combo_count(hg, mask)
        ratio = L / G if G else float("inf")
        rows_rep.append(dict(screen=tag, mask=mask, loss_px=L, gain_px=G,
                             loss_ha=round(ha(L), 1), gain_ha=round(ha(G), 1),
                             loss_per_reverse=round(ratio, 2) if G else None,
                             kept_pct=round(100 * L / max(tot_loss, 1), 1)))
        print(f"{tag:<26}{L:>12,}{ha(L):>11,.0f}{G:>12,}{ha(G):>10,.0f}"
              f"{ratio:>10.2f}{100*L/max(tot_loss,1):>7.1f}%")

    add("none (raw)", 0)
    avail = {"SINGLE": r["have"]["ss"], "CFWD": r["have"]["ins"], "CREV": r["have"]["ins"],
             "PROB": r["have"]["pc"], "SPATIAL": True, "PLAUS": True}
    for s in SCREENS:
        if avail[s]:
            add(s, BIT[s])
    print(f"{'-'*94}")
    # Curated combinations rather than every one of the 63: the cumulative
    # prefix of an arbitrary screen order is not an interesting family, and it
    # drags along any screen that turns out to be anti-informative on its own.
    for combo in (("PLAUS", "SPATIAL"), ("CFWD", "PLAUS"), ("CFWD", "SPATIAL", "PLAUS"),
                  ("SINGLE", "SPATIAL", "PLAUS"), ("CFWD", "PROB", "PLAUS"),
                  ("CFWD", "PROB", "SPATIAL", "PLAUS"),
                  ("SINGLE", "CFWD", "PROB", "SPATIAL", "PLAUS")):
        if all(avail[c] for c in combo):
            add("+".join(combo), sum(BIT[c] for c in combo))
    print(f"{'-'*94}")
    if args.screen_mask and args.screen_mask not in [x["mask"] for x in rows_rep]:
        add("SELECTED: " + "+".join(names), args.screen_mask)
    print(f"{'='*94}")
    print("loss:rev of 1.0 means the layer is indistinguishable from classifier "
          "variance;\na screen is only worth its cost if it raises that ratio. But "
          "read the ratio\nwith the asymmetry in mind: PROB and CFWD test confidence "
          "in the DESTINATION\nclass, and settlement/cropland are far easier classes "
          "than mire or sparse veg,\nso those two screens flatter themselves in the "
          "forward direction. SINGLE, SPATIAL\nand PLAUS treat both directions alike "
          "and their ratios are the honest ones.")

    print(f"\nnature loss by source class (SELECTED screen, before MMU):")
    for c in args.natural:
        if r["from_raw"][c]:
            print(f"  {c:>3} {labels.get(c,'?'):<24} raw {r['from_raw'][c]:>10,} px "
                  f"({ha(r['from_raw'][c]):>9,.0f} ha)   screened "
                  f"{r['from_scr'][c]:>9,} px ({ha(r['from_scr'][c]):>8,.0f} ha)")
    print(f"\nnature loss by destination:")
    for c, n in args.anthro.items():
        print(f"  {c:>3} {n:<24} raw {r['to_raw'][c]:>10,} px "
              f"({ha(r['to_raw'][c]):>9,.0f} ha)   screened "
              f"{r['to_scr'][c]:>9,} px ({ha(r['to_scr'][c]):>8,.0f} ha)")

    print(f"\napplying MMU >= {args.mmu_px} px ({ha(args.mmu_px):.2f} ha) on top of "
          f"{'+'.join(names) or 'NONE'} ...", flush=True)
    mmu = apply_mmu(r["out"], args.mmu_px, args.screen_mask)
    lk, rk = mmu["loss_kept_px"], mmu["reverse_kept_px"]
    print(f"  tier 4 (HIGH) nature loss: {lk:,} px ({ha(lk):,.0f} ha) in "
          f"{mmu['loss_patches']:,} patches")
    print(f"    dropped as sub-MMU speckle: {mmu['loss_dropped_px']:,} px "
          f"({ha(mmu['loss_dropped_px']):,.0f} ha) over "
          f"{mmu['loss_components'] - mmu['loss_patches']:,} components")
    tail = f"  ->  loss:rev {lk/rk:.2f}" if rk else "  ->  control is empty"
    print(f"  reverse control at the same MMU: {rk:,} px ({ha(rk):,.0f} ha) in "
          f"{mmu['reverse_patches']:,} patches{tail}")
    mmu["loss_per_reverse"] = round(lk / rk, 2) if rk else None

    rep = {"kind": "nature_loss", "years": [y0, y1], "raster": str(r["out"]),
           "natural": list(args.natural), "anthropogenic": args.anthro,
           "grass_to_crop_included": args.include_grass_to_crop,
           "selected_screen": names, "nbr": args.nbr, "mmu_px": args.mmu_px,
           "p_hi": args.p_hi, "p_lo": args.p_lo,
           "bands": {"1": "loss_tier 0=nodata 1=no loss 2=candidate "
                          "3=passes screen 4=passes screen+MMU",
                     "2": "from_class",
                     "3": "screen bitmask; bit 6 (64) marks a reverse-direction "
                          "control pixel, not a screen"},
           "screen_bits": {n: BIT[n] for n in SCREENS},
           "valid_px": r["n_valid"], "raw_loss_px": tot_loss, "raw_reverse_px": tot_gain,
           "screen_comparison": rows_rep,
           "from_class_raw": r["from_raw"], "from_class_screened": r["from_scr"],
           "to_class_raw": r["to_raw"], "to_class_screened": r["to_scr"],
           "mmu": mmu, "ha_per_px": HA_PER_PX,
           "reverse_bit": REVERSE_BIT,
           "caveat": "the reverse flow (anthropogenic -> natural) is the "
                     "false-positive control; screens that test confidence in the "
                     "destination class (PROB, CFWD) look better than they are "
                     "because settlement/cropland are easier classes than the "
                     "natural ones"}
    j = d / f"nature_loss{args.tag_part}_{y0}_{y1}.json"
    j.write_text(json.dumps(rep, indent=2, default=str))
    print(f"\n  report -> {j}")


if __name__ == "__main__":
    main()
