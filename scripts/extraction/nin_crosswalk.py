"""Crosswalk NiN_v2 naturtyper -> FSCS 12-class legend.

NiN_v2 (`P:/154001_nyvest/GIS/Naturbase/NiN_v2/NiN_v2_nyvest.shp`) is the
field-mapped "Natur i Norge" type-2 nature-type inventory for the AOI: 28,496
polygons, surveyed 2018-2023, EPSG:25833. It is more recent and higher-certainty
than grunnkart, so it is a candidate source of extra training labels for the
confusable VEGETATION classes (cls4 forest, cls5 grassland, cls6 scrub/heath,
cls7 wetland). It has NO snow/ice category, so it cannot help class 12.

FSCS legend (see schemeB-class12 memory):
  2=bare  3=cropland  4=forest  5=grassland  6=scrub/heathland  7=wetland
  8=water  10=settlement  11=infrastructure  12=snow/ice

Mapping strategy: the NiN `hovedøkosy` (main ecosystem) gives the first-order
class; `naturtype` keyword overrides handle the cases where one ecosystem spans
two FSCS classes (e.g. semi-naturligMark contains both grassland and dwarf-shrub
heathland; naturligÅpneOmråder contains both bare rock/sand and coastal meadow).

A naturtype is mapped to None (dropped) when it is not a defensible land-cover
label at Sentinel-2 (10 m) scale — e.g. Hule eiker (single hollow oaks).
"""
from __future__ import annotations

# FSCS class codes
BARE, CROP, FOREST, GRASS, SCRUB, WETLAND, WATER, SETTLE, INFRA, SNOW = (
    2, 3, 4, 5, 6, 7, 8, 10, 11, 12)

# Default FSCS class per NiN hovedøkosy (main ecosystem).
ECOSYSTEM_DEFAULT = {
    "skog": FOREST,
    "semi-naturligMark": GRASS,            # overridden to SCRUB for heath types
    "våtmark": WETLAND,
    "fjell": SCRUB,                        # alpine heath/tundra; vegetated
    "naturligÅpneOmråderILavlandet": GRASS,  # overridden to BARE for rock/sand
}

# naturtype keyword -> FSCS class override (checked case-insensitively as a
# substring; first match wins, order matters). These resolve the ecosystems
# that straddle two FSCS classes.
KEYWORD_OVERRIDES = [
    # --- forested wetland stays wetland even though name contains 'skog' ---
    ("sumpskog", WETLAND),
    ("kildeskogsmark", WETLAND),
    ("kildelauvskog", WETLAND),
    ("strandskog", WETLAND),
    ("kildeedellauvskog", WETLAND),
    # --- heathland / dwarf-shrub within semi-naturligMark & fjell -> SCRUB ---
    ("lynghei", SCRUB),
    ("kystlynghei", SCRUB),
    ("boreal hei", SCRUB),
    ("fjellhei", SCRUB),
    ("rasmarkhei", SCRUB),
    # --- bare rock / sand / scree within naturligÅpneOmråder & fjell -> BARE -
    ("berg", BARE),       # *kalkberg, fosseberg, snøleieberg
    ("blokkmark", BARE),  # snøleieblokkmark (boulder field)
    ("rabbe", BARE),      # exposed alpine ridge
    ("sanddyne", BARE),
    ("sanddynemark", BARE),
    ("grunnlendt", BARE),  # åpen grunnlendt kalkrik mark (shallow-soil rock)
    # --- drop: not land cover at 10 m ---
    ("hule eiker", None),
]


def map_naturtype(naturtype: str, hovedokosy: str):
    """Return the FSCS class code for a NiN polygon, or None to drop it."""
    nt = (naturtype or "").strip().lower()
    for kw, cls in KEYWORD_OVERRIDES:
        if kw in nt:
            return cls
    return ECOSYSTEM_DEFAULT.get(hovedokosy)


if __name__ == "__main__":
    # Self-check: print the resulting class distribution over the QC'd polygons.
    import geopandas as gpd
    import pandas as pd

    SRC = ("/data/P-Prosjekter2/154001_nyvest/GIS/Naturbase/NiN_v2/"
           "NiN_v2_nyvest.shp")
    g = gpd.read_file(
        SRC, columns=["hovedøkosy", "naturtype", "usikkerhet", "area_m2"])
    qc = g[(g["usikkerhet"] != "Ja")
           & (g["naturtype"] != "Hule eiker")
           & (g["area_m2"] >= 900)].copy()
    qc["fscs"] = [map_naturtype(nt, eco) for nt, eco
                  in zip(qc["naturtype"], qc["hovedøkosy"])]
    print(f"QC polygons: {len(qc)}  (dropped to None: {qc['fscs'].isna().sum()})")
    print("\nFSCS class distribution (polygon count, total km2):")
    summ = (qc.dropna(subset=["fscs"])
            .groupby("fscs")
            .agg(n=("area_m2", "size"), km2=("area_m2", lambda s: s.sum() / 1e6))
            .sort_index())
    print(summ.to_string())
    # show which naturtypes land in each class for review
    print("\nnaturtypes per FSCS class:")
    pd.set_option("display.max_rows", 300)
    for cls in sorted(qc["fscs"].dropna().unique()):
        nts = qc[qc["fscs"] == cls]["naturtype"].value_counts()
        print(f"\n--- FSCS {int(cls)} ({nts.sum()} polys) ---")
        print(nts.head(12).to_string())
