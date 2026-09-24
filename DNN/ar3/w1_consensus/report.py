"""Step 4 — builds data/consensus/README.md from the finished pipeline
outputs: candidates.parquet / points.parquet, reference_points.parquet, the
three validation-check numbers (step2_validate), and product-agreement /
gk_plus2-survival statistics computed here via crosswalk.py.

Purely descriptive — no modelling recommendations (per the brief).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from common import CONSENSUS_DIR
from crosswalk import to_merged, consensus, GRUNNKART_MAP

POINTS_PARQUET = os.path.join(CONSENSUS_DIR, "points.parquet")
CANDIDATES_PARQUET = os.path.join(CONSENSUS_DIR, "candidates.parquet")
REFERENCE_PARQUET = os.path.join(CONSENSUS_DIR, "reference_points.parquet")
AEF_LONG_PARQUET = os.path.join(CONSENSUS_DIR, "aef_long.parquet")
VALIDATION_JSON = os.path.join(CONSENSUS_DIR, "_validation_checks.json")
OUT_README = os.path.join(CONSENSUS_DIR, "README.md")

PRODUCT_COLS = {
    "worldcover": None,  # handled specially (wc2020/wc2021, no year loop)
    "esri": "esri_{y}",
    "dw": "dw_{y}",
}
REPORT_YEARS = (2018, 2024)


def _product_merged(df, product, year, gk_col):
    if product == "worldcover":
        from crosswalk import _nearest_worldcover_year
        col = f"wc{_nearest_worldcover_year(year)}"
    else:
        col = PRODUCT_COLS[product].format(y=year)
    if col not in df.columns:
        return None
    return to_merged(product, df[col])


def agreement_table(df, gk_col="gk_v1", years=REPORT_YEARS, group_col=None):
    """Fraction of rows where each product's merged class is compatible
    with `gk_col`'s merged class, among rows where both are defined.
    Returns a long DataFrame: group, product, year, n_both_defined, n_agree,
    agreement_rate."""
    gk_merged = to_merged("grunnkart", df[gk_col])
    groups = df[group_col] if group_col else pd.Series(["all"] * len(df))
    rows = []
    for product in ("worldcover", "esri", "dw"):
        for y in years:
            ext = _product_merged(df, product, y, gk_col)
            if ext is None:
                continue
            for g in sorted(groups.unique()):
                idx = np.flatnonzero((groups == g).to_numpy())
                n_both = n_agree = 0
                for i in idx:
                    gm, em = gk_merged[i], ext[i]
                    if gm is None or em is None:
                        continue
                    n_both += 1
                    compat = (gm in em) if isinstance(em, (set, frozenset)) else (gm == em)
                    if compat:
                        n_agree += 1
                rate = n_agree / n_both if n_both else float("nan")
                rows.append({"group": g, "product": product, "year": y,
                            "n_both_defined": n_both, "n_agree": n_agree,
                            "agreement_rate": rate})
    return pd.DataFrame(rows)


def survival_table(df, gk_col="gk_v1", years=REPORT_YEARS, group_col="stratum"):
    rows = []
    groups = df[group_col] if group_col in df.columns else pd.Series(["all"] * len(df))
    for y in years:
        labels, n_agree, n_disagree = consensus(df, y, rule="gk_plus2", gk_col=gk_col)
        survived = labels != -1
        gk_merged = to_merged("grunnkart", df[gk_col])
        for g in sorted(groups.unique()):
            idx = (groups == g).to_numpy()
            n_tot = int(idx.sum())
            n_surv = int(survived[idx].sum())
            rows.append({"group": g, "year": y, "n_total": n_tot,
                        "n_survive": n_surv,
                        "survive_frac": n_surv / n_tot if n_tot else float("nan")})
        # per merged-class survival (using gk_col's own merged class as the class label)
        for c in sorted(set(v for v in gk_merged if v is not None)):
            idx = np.array([v == c for v in gk_merged])
            n_tot = int(idx.sum())
            n_surv = int(survived[idx].sum())
            rows.append({"group": f"class={c}", "year": y, "n_total": n_tot,
                        "n_survive": n_surv,
                        "survive_frac": n_surv / n_tot if n_tot else float("nan")})
    return pd.DataFrame(rows)


def md_table(df, float_cols=()):
    df = df.copy()
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
            "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


WALL_TIME_SUMMARY = """
- Step 1 (candidate sampling, local rasters only): 8s for all 48,000
  candidates (single round; each of the 312 grid cells with any raster
  block contributed once, matched to its target in that one pass).
- Step 2 local (grunnkart v1/v2 + lidar, 48,000 points): ~90s, folded into
  the first launch attempt's log.
- Step 2 GEE static extraction (WorldCover + Esri + Dynamic World,
  2017-2025) + temporal AEF extraction (2017-2025), 48,000 candidate
  points, run concurrently (10-worker thread pool): 4,075s (~68 min),
  timed by the script itself from just after local extraction to the
  final flush.
- Step 2 reference sample (3,000 points, static products only): completed
  well inside the first ~15 min of the run.
- Validation checks (cell_id / lidar / AEF-scaling, 500+500+300 points):
  under 2 minutes.
- Total wall clock from first launch to all outputs on disk: ~80 minutes
  (14:34-15:54), which INCLUDES an aborted sequential-extraction attempt
  and a mid-run rewrite to add concurrency (see Surprises) — the clean
  68-minute figure above is what a rerun from a warm local-extraction
  cache would take today.
""".strip()

SURPRISES_SUMMARY = """
- **Sequential extraction was far too slow; rewrote to concurrent
  mid-run.** The first implementation issued one `computeFeatures` call at
  a time. On the 200-point end-to-end test this looked fine (~12s/call),
  but at the real batch size (2,000 points) each call took 60-150s, and
  with 240 static + 216 temporal subtasks that projected to many hours.
  Rewrote `gee_extract.py` to dispatch pending subtasks across a
  10-worker `ThreadPoolExecutor` (matching
  `scripts/extraction/sample_feature_space_stable_allyears.py`'s own
  concurrency pattern, which the sequential-first draft had NOT copied
  despite the brief pointing at that script). This cut per-subtask time to
  ~8s on average — an ~8-10x speedup — bringing the full run down to
  ~68 minutes. No data was lost: the partially-completed sequential run's
  checkpoint/parquet state was reused by the concurrent version.
- **A real checkpoint-resume bug was caught before it corrupted data.**
  `run_static`'s DuckDB buffer started every process run as a 1-column
  (`pid`-only) skeleton table; on a restart with existing multi-column
  data already on disk, `INSERT ... SELECT *` from that parquet into the
  1-column table raised a column-count mismatch. Fixed by re-adding the
  on-disk columns before inserting, and verified with an isolated
  restart test (partial run -> new process -> more years) before
  relaunching the real job.
- **`ESA/WorldCover/v100`/`v200` are single-image ImageCollections, not
  bare Image assets** — `ee.Image(WORLDCOVER_V100)` fails with
  "Image.load: ... is not an Image"; caught by the 200-point test, fixed
  to `ee.Image(ee.ImageCollection(...).first())`.
- **A crosswalk correctness bug was caught by a synthetic unit check, not
  real data.** The first cut of `to_merged()` let ANY product's raw code
  fall back to "identity" if it numerically coincided with one of the 9
  merged-class ids — meaning e.g. a WorldCover code of `4` (not a real
  WorldCover code) would have silently been read as merged class 4
  (forest). Restricted the identity fallback to grunnkart only, whose raw
  code space genuinely IS the merged-class numbering (that's the brief's
  own "others identity" instruction) — WorldCover/Esri/DW codes not in
  their explicit dict now map to `None`.
- **Step-1 candidate spreading needed a round-robin fix.** The first
  accept-order (first-come-first-served across a thread pool) let
  whichever cells' blocks finished first exhaust the global per-stratum
  target, leaving candidates concentrated in only 72 of ~300 candidate
  cells. Switched to round-robin allocation across cells within each
  round; final spread is 124-144 distinct cells per stratum (flip max
  428 vs median 212 per cell — still uneven, since it is genuinely
  bounded by how much flip/uncertain-eligible area each cell has, per the
  brief's own "a cell with little area ... contributes what it has").
- **Esri LULC 2025 coverage is a different (untiled) image than
  2017-2024.** The 2017-2024 images for this AOI carry an `id_no` property
  (`32V_<year>`, one UTM-zone tile); the 2025 image has no `id_no` — it is
  a larger, untiled global mosaic — but does cover the AOI (confirmed by
  direct `reduceRegion` histogram sampling). Recorded as available for all
  9 years; no fallback needed.
- **1/48,000 candidate points (pid 45787) has zero valid AEF rows across
  all 9 years** — every `sampleRegions` call for that point returned
  nothing. Left as a real null (not imputed); negligible at this rate.
- **Two candidate points share the same 30 m spacing bin** under an
  independent re-check (round-trip EPSG:4326<->32633 reprojection can flip
  a boundary point's bin at the sub-cm level even though the enforcement
  at accept-time was correct in the raster's native EPSG:32633 grid) — 2
  of 48,000 pairs, not a real violation of the >=30 m intent.
- **`gk_plus2` survival is very low for wetland and bare in the hard-area
  strata** (class 7: 0.4%/0.2% for 2018/2024; class 2: 3.7%/6.8%) — i.e.
  grunnkart's own label for those classes rarely gets 2 external products
  to agree AND outvote disagreement, in these specific flip/uncertain/
  random pixels. This is reported as a fact about the crosswalk/data, not
  a recommendation.
- Report defaults to `gk_v1` (matching the same raster the training
  parquet's `class` field and the GEE training asset both derive from) as
  "the" grunnkart label for agreement/survival stats; `gk_v2` is also in
  `points.parquet`/`reference_points.parquet` and `report.build()` takes
  `gk_col` as a parameter if the harness wants the other version instead.
""".strip()


def build(gk_col="gk_v1", wall_time_summary=WALL_TIME_SUMMARY,
         surprises=SURPRISES_SUMMARY):
    points = pd.read_parquet(POINTS_PARQUET)
    candidates = pd.read_parquet(CANDIDATES_PARQUET)
    reference = (pd.read_parquet(REFERENCE_PARQUET)
                if os.path.exists(REFERENCE_PARQUET) else None)
    validation = (json.load(open(VALIDATION_JSON))
                 if os.path.exists(VALIDATION_JSON) else {})
    aef_n = (len(pd.read_parquet(AEF_LONG_PARQUET, columns=["pid"]))
            if os.path.exists(AEF_LONG_PARQUET) else 0)

    stratum_counts = candidates["stratum"].value_counts().rename_axis("stratum") \
        .reset_index(name="n")
    cell_hist = (candidates.groupby("stratum")["cell_id"]
                .agg(n_cells="nunique", min_per_cell=lambda s: s.value_counts().min(),
                     median_per_cell=lambda s: s.value_counts().median(),
                     max_per_cell=lambda s: s.value_counts().max())
                .reset_index())
    class_counts = (candidates.groupby(["stratum", "class_2018", "class_2024"])
                    .size().reset_index(name="n")
                    .sort_values(["stratum", "n"], ascending=[True, False]))

    agree_cand = agreement_table(points, gk_col=gk_col, group_col="stratum")
    agree_ref = (agreement_table(reference, gk_col=gk_col, group_col=None)
                if reference is not None else pd.DataFrame())
    survive = survival_table(points, gk_col=gk_col, group_col="stratum")

    lines = []
    lines.append("# data/consensus — W1 consensus-labels dataset\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M')} "
                 f"(local server time). Built per "
                 f"`DNN/ar3/briefs/phase1_consensus.md`; code in "
                 f"`DNN/ar3/w1_consensus/`.\n")

    lines.append("## Files\n")
    lines.append(f"- `candidates.parquet`: {len(candidates):,} rows — "
                 f"stratum, cell_id, lon, lat, class_2018, class_2024, "
                 f"setsize_2018, setsize_2024 (step 1 output)")
    lines.append(f"- `points.parquet`: {len(points):,} rows, "
                 f"{points.shape[1]} columns — candidates + gk_v1/gk_v2 + "
                 f"lidar + wc2020/wc2021 + esri_<year>/dw_<year>[_frac,_n] "
                 f"2017-2025")
    lines.append(f"- `aef_long.parquet`: {aef_n:,} rows — pid, year, "
                 f"A00..A63 for 2017-2025 (candidate points only)")
    if reference is not None:
        lines.append(f"- `reference_points.parquet`: {len(reference):,} "
                     f"rows — same product extraction on existing training "
                     f"points, stratified by class\n")
    else:
        lines.append("- `reference_points.parquet`: NOT PRODUCED (see "
                     "Problems)\n")

    lines.append("## Counts per stratum\n")
    lines.append(md_table(stratum_counts))
    lines.append("\n## Per-cell histogram (candidates)\n")
    lines.append(md_table(cell_hist, float_cols=["median_per_cell"]))
    lines.append("\n## Class-transition counts per stratum "
                 "(class_2018 -> class_2024, top rows)\n")
    lines.append(md_table(class_counts.groupby("stratum").head(10)))

    lines.append(f"\n## Product agreement with grunnkart ({gk_col})\n")
    lines.append("### By stratum (candidate/hard-area points)\n")
    lines.append(md_table(agree_cand, float_cols=["agreement_rate"]))
    if not agree_ref.empty:
        lines.append("\n### On the reference sample "
                     "(3,000 existing training points, clean frame)\n")
        lines.append(md_table(agree_ref.drop(columns=["group"]),
                              float_cols=["agreement_rate"]))

    lines.append(f"\n## Survival fraction under `gk_plus2` (vs {gk_col})\n")
    lines.append(md_table(survive, float_cols=["survive_frac"]))

    lines.append("\n## Validation checks\n")
    if validation:
        c1 = validation.get("cell_id", {})
        lines.append(f"1. **cell_id**: {c1.get('n_match')}/{c1.get('n')} "
                     f"existing training points' recomputed cell_id matched "
                     f"the parquet's own cell_id "
                     f"(all_match={c1.get('all_match')}).")
        c2 = validation.get("lidar", {})
        lines.append(f"2. **lidar**: re-extracted "
                     f"{c2.get('n_matched_in_ref')}/{c2.get('n_sampled')} "
                     f"existing points found in data/lidar_features.parquet; "
                     f"max |Δ| per column: "
                     f"{json.dumps(c2.get('max_abs_delta'))}.")
        c3 = validation.get("aef_scaling", {})
        lines.append(f"3. **AEF scaling**: max |Δ| vs the training parquet, "
                     f"per year: {json.dumps(c3)}.")
    else:
        lines.append("Validation JSON not found — see Problems.")

    lines.append("\n## Wall time\n")
    lines.append(wall_time_summary or "(see logs/ timestamps)")

    lines.append("\n## Surprises / deviations from the brief\n")
    lines.append(surprises or "(none recorded)")

    with open(OUT_README, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[OK] wrote {OUT_README}")


if __name__ == "__main__":
    build()
