"""Crosswalk from each product's native class codes onto the 9 merged
classes used by the deployed model (2 bare, 3 crop, 4 forest, 5 grassland,
6 scrub, 7 wetland, 8 water, 10 built, 12 snow/ice).

This module ONLY maps codes and computes the vote-based consensus label; it
deliberately does NOT pick a single "final" rule — `consensus()` exposes
`gk_plus2`, `ext_majority` and `gk_raw` so a downstream harness can run all
three as controls (see DNN/ar3/PLAN.md, workstream W3).

Verified against the actual products (2026-09-23, project ee-gsingh):

grunnkart (rasterized_10m/grunnkart_nyvest_10m[_v2].tif, raw class 1..13):
  matches DNN/README.md's class merge (1->2, 9->8) and
  landcover_2018_2024/MANIFEST.md's legend (2 bare, 10 built, no 1/9/11 —
  11 sparse-veg is folded into 2 per PLAN.md MERGE_EXTRA=11:2). 13 has no
  merged-class counterpart -> None.

ESA WorldCover (`ESA/WorldCover/v100` = 2020, `v200` = 2021, band `Map`):
  standard 11-class legend (10 tree cover .. 100 moss/lichen); 95 mangroves
  does not occur in Norway -> None.

Dynamic World (`GOOGLE/DYNAMICWORLD/V1`, band `label`): confirmed band order
  via .bandNames() = [water, trees, grass, flooded_vegetation, crops,
  shrub_and_scrub, built, bare, snow_and_ice, label] -> label values 0..8 in
  that same order.

Esri 10 m LULC time series
  (`projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS`,
  band `b1`): confirmed by sampling the AOI — values {1,2,4,5,7,8,9,11}
  present (2,4,5,7,8,9,11 = trees/flooded-veg/crops/built/bare/snow/
  rangeland; 10=clouds not observed but mapped anyway). Coverage for the
  AOI: tiled per-UTM-zone images '32V_<year>' for 2017-2024 (id_no
  property), PLUS an untiled global mosaic image for 2025 (no id_no, but
  covers the AOI — confirmed via reduceRegion histogram) -> all 9 years
  2017-2025 are available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── raw-code -> merged-class dictionaries ──────────────────────────
# int -> (int merged class) | None | frozenset (ambiguous / compatible-with-either)
GRUNNKART_MAP = {1: 2, 9: 8, 11: 2, 13: None}
# all other raw codes (2,3,4,5,6,7,8,10,12) map to themselves (identity)

WORLDCOVER_MAP = {
    10: 4, 20: 6, 30: 5, 40: 3, 50: 10, 60: 2, 70: 12, 80: 8, 90: 7,
    95: None, 100: 2,
}

DYNAMICWORLD_MAP = {
    0: 8,   # water
    1: 4,   # trees
    2: 5,   # grass
    3: 7,   # flooded_vegetation
    4: 3,   # crops
    5: 6,   # shrub_and_scrub
    6: 10,  # built
    7: 2,   # bare
    8: 12,  # snow_and_ice
}

ESRI_MAP = {
    1: 8,    # water
    2: 4,    # trees
    4: 7,    # flooded vegetation
    5: 3,    # crops
    7: 10,   # built area
    8: 2,    # bare ground
    9: 12,   # snow/ice
    10: None,            # clouds
    11: frozenset({5, 6}),  # rangeland: compatible with grassland OR scrub
}

MERGED_CLASSES = frozenset({2, 3, 4, 5, 6, 7, 8, 10, 12})

_PRODUCT_MAPS = {
    "grunnkart": GRUNNKART_MAP,
    "gk": GRUNNKART_MAP,
    "worldcover": WORLDCOVER_MAP,
    "wc": WORLDCOVER_MAP,
    "dw": DYNAMICWORLD_MAP,
    "dynamicworld": DYNAMICWORLD_MAP,
    "esri": ESRI_MAP,
}


def to_merged(product, codes):
    """Map raw `codes` (array-like of numbers, NaN allowed) from `product`
    onto the 9 merged classes.

    Returns a numpy object array, one entry per input code:
      - int in MERGED_CLASSES  : unambiguous merged class
      - frozenset of ints      : compatible with any class in the set
                                  (currently only Esri rangeland -> {5,6})
      - None                   : no merged-class counterpart / missing code
                                  (also used for NaN/unmapped raw codes)

    `product` is looked up case-insensitively. Grunnkart's raw code space
    IS the merged-class numbering (that's why the brief says "others
    identity" for it) so any grunnkart code not in GRUNNKART_MAP but
    already a merged class passes through unchanged. WorldCover/Esri/DW use
    an UNRELATED numbering, so for those products a code missing from the
    dict maps to None rather than an identity fallback — a WorldCover code
    of e.g. 4 is not "forest" just because 4 happens to be forest's merged
    id, it is an invalid/unexpected code and is dropped.
    """
    key = str(product).strip().lower()
    if key not in _PRODUCT_MAPS:
        raise ValueError(f"Unknown product {product!r}; "
                         f"expected one of {sorted(set(_PRODUCT_MAPS))}")
    mapping = _PRODUCT_MAPS[key]
    identity_fallback = key in ("grunnkart", "gk")

    codes = pd.Series(codes) if not isinstance(codes, pd.Series) else codes
    out = np.empty(len(codes), dtype=object)
    for i, c in enumerate(codes.to_numpy()):
        if c is None or (isinstance(c, float) and np.isnan(c)):
            out[i] = None
            continue
        ci = int(c)
        if ci in mapping:
            out[i] = mapping[ci]
        elif identity_fallback and ci in MERGED_CLASSES:
            out[i] = ci  # grunnkart only: codes already in the merged scheme
        else:
            out[i] = None
    return out


def _is_compatible(external_val, gk_val):
    if external_val is None or gk_val is None:
        return None  # no information -> excluded from the vote
    if isinstance(external_val, (set, frozenset)):
        return gk_val in external_val
    return external_val == gk_val


def _nearest_worldcover_year(year):
    return 2020 if abs(year - 2020) <= abs(year - 2021) else 2021


def consensus(points_df, year, rule="gk_plus2", gk_col="gk_v1"):
    """Per-row consensus label against `gk_col` for calendar `year`.

    External products consulted: WorldCover (nearest of 2020/2021 to
    `year`), Esri_<year> (column `esri_<year>`), Dynamic World_<year>
    (column `dw_<year>`). A product whose column is absent or whose value
    maps to None for that row is excluded from the vote (neither agree nor
    disagree) rather than counted as a disagreement.

    rule:
      "gk_plus2"    keep `gk_col`'s merged label iff >=2 of the (up to 3)
                    external products are compatible with it AND
                    n_agree > n_disagree.
      "gk_raw"      always keep `gk_col`'s merged label (no external
                    filter) — control for "labels in hard areas" alone.
      "ext_majority" plurality vote among the external products only,
                    ignoring `gk_col` entirely (grunnkart-free control);
                    ambiguous (set-valued) external votes are excluded
                    from the tally since they can't cast a single vote;
                    ties -> no label (-1).

    Returns (labels, n_agree, n_disagree): three int64 arrays, one row per
    input row. labels[i] == -1 means "no consensus label for this rule".
    n_agree/n_disagree are always computed against `gk_col` (even under
    "ext_majority") so callers can inspect agreement regardless of rule.
    """
    n = len(points_df)
    gk_raw_codes = points_df[gk_col].to_numpy()
    gk_merged = to_merged("grunnkart", gk_raw_codes)

    wc_year = _nearest_worldcover_year(year)
    wc_col = f"wc{wc_year}"
    wc_merged = (to_merged("worldcover", points_df[wc_col])
                if wc_col in points_df.columns else np.array([None] * n, dtype=object))

    esri_col = f"esri_{year}"
    esri_merged = (to_merged("esri", points_df[esri_col])
                  if esri_col in points_df.columns else np.array([None] * n, dtype=object))

    dw_col = f"dw_{year}"
    dw_merged = (to_merged("dw", points_df[dw_col])
                if dw_col in points_df.columns else np.array([None] * n, dtype=object))

    externals = [wc_merged, esri_merged, dw_merged]

    labels = np.full(n, -1, dtype=np.int64)
    n_agree = np.zeros(n, dtype=np.int64)
    n_disagree = np.zeros(n, dtype=np.int64)

    for i in range(n):
        g = gk_merged[i]
        a = d = 0
        if g is not None:
            for e in externals:
                comp = _is_compatible(e[i], g)
                if comp is None:
                    continue
                if comp:
                    a += 1
                else:
                    d += 1
        n_agree[i] = a
        n_disagree[i] = d

        if rule == "gk_raw":
            if g is not None:
                labels[i] = g
        elif rule == "gk_plus2":
            if g is not None and a >= 2 and a > d:
                labels[i] = g
        elif rule == "ext_majority":
            votes = {}
            for e in externals:
                ev = e[i]
                if ev is None or isinstance(ev, (set, frozenset)):
                    continue
                votes[ev] = votes.get(ev, 0) + 1
            if votes:
                top = max(votes.values())
                winners = [k for k, v in votes.items() if v == top]
                if len(winners) == 1:
                    labels[i] = winners[0]
        else:
            raise ValueError(f"Unknown rule {rule!r}")

    return labels, n_agree, n_disagree
