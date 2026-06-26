"""Active-learning labeling app for Norwegian nature types.

Streamlit + Folium. Seeded by the fast CatBoost model (GPU-trained) + APS conformal
uncertainty (macro-F1 ≈ 0.69 on spatial holdout; the heavy TabICL pipeline reaches ≈0.707
and is available as an offline high-quality map pass). The same APS prediction-set size is
both the map's uncertainty layer AND the active-learning acquisition score.

Loop: the queue surfaces the most uncertain (largest prediction-set) unlabeled
locations → you inspect a point (its prediction set, grunnkart label, probabilities)
→ assign a class → "Retrain" refits CatBoost (GPU) with your labels folded in and rescores
every location in seconds → the map + queue update. "Regenerate map" reruns the dense
single-tile raster with the current model.

Run (on the A40 server, then port-forward 8501):
    ~/myprojects/recover/.venv/bin/python -m streamlit run active_learning/scripts/app.py \
      --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_core as mc
from build_predictions import build_predictions, PRED_PARQUET

LABELS_PARQUET = mc.AL_DATA_DIR / "labels.parquet"
RASTER_DIR = mc.AL_ROOT / "reports"

# class code -> hex colour (10 distinct, colourblind-leaning). Keyed by merged code.
CLASS_COLORS = {
    2: "#8c8c8c",   # bare
    3: "#e6c200",   # cropland
    4: "#1b7837",   # forest
    5: "#a6d96a",   # grassland
    6: "#d9a066",   # scrub/heathland
    7: "#5ab4ac",   # wetland
    8: "#2166ac",   # water
    10: "#d73027",  # settlement
    11: "#762a83",  # infrastructure
    12: "#f7f7f7",  # snow/ice
}

st.set_page_config(page_title="Nature-type Active Learning", layout="wide")


# ==============================================================================
# Cached resources / state
# ==============================================================================
@st.cache_resource(show_spinner=False)
def _seed_holder():
    """Mutable single-element holder so a retrain can swap the live model in place
    without busting other caches."""
    return {"model": mc.load_seed_model(), "version": 0}


def get_seed() -> mc.SeedModel:
    return _seed_holder()["model"]


def load_labels() -> pd.DataFrame:
    if LABELS_PARQUET.exists():
        return pd.read_parquet(LABELS_PARQUET)
    return pd.DataFrame(columns=["lon", "lat", "label", "source", "ts"])


def save_label(lon: float, lat: float, label: int):
    df = load_labels()
    # one label per location: drop any prior label at (lon,lat), then append.
    df = df[~((df.lon == lon) & (df.lat == lat))]
    row = {"lon": lon, "lat": lat, "label": int(label),
           "source": "user", "ts": time.time()}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    LABELS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LABELS_PARQUET, index=False)


@st.cache_data(show_spinner=False)
def load_predictions(version: int) -> pd.DataFrame:
    """Predictions table (cached on the model version so a retrain refreshes it)."""
    if not PRED_PARQUET.exists():
        return build_predictions(get_seed())
    return pd.read_parquet(PRED_PARQUET)


@st.cache_data(show_spinner=False)
def list_rasters():
    cls = sorted(RASTER_DIR.glob("*_class_3857.tif"))
    return cls


# ==============================================================================
# Live retrain
# ==============================================================================
@st.cache_resource(show_spinner=False)
def _stable_cache():
    """Deduped stable frame as arrays, loaded once (≈74k×64). Reused every retrain so
    the loop never re-reads the 663k-row parquet."""
    df = mc.dedup_to_latest_year(mc.load_stable())
    return {
        "X": df[mc.FEATURE_COLS].values.astype(np.float32),
        "y": mc.merge_classes(df[mc.TARGET].values).astype(int),
        "lonlat": df[["lon", "lat"]].values.astype(np.float64),
    }


def _fold_user_labels(y, lonlat, labels, classes, extra_weight):
    """Vectorised: override labels at user-labelled locations + return sample weights."""
    y = y.copy()
    weights = np.ones(len(y), dtype=np.float64)
    if not len(labels):
        return y, weights
    valid = set(int(c) for c in classes)
    # round to a stable key and match by dict lookup (vectorised over the small label set)
    keys = np.round(lonlat, 8)
    lut = {(round(float(r.lon), 8), round(float(r.lat), 8)): int(r.label)
           for r in labels.itertuples() if int(r.label) in valid}
    for j in range(len(y)):
        k = (keys[j, 0], keys[j, 1])
        lab = lut.get(k)
        if lab is not None:
            y[j] = lab
            weights[j] = extra_weight
    return y, weights


def retrain_live(mode: str = "fast", extra_weight: float = 5.0):
    """Refit CatBoost on stable rows + user labels (upweighted), recalibrate APS tau,
    swap the live model, rebuild predictions.

    mode='fast'  : ~15k working rows, 150 iters, single-split calibration (~10s).
                   τ barely moves for a handful of labels, so this stays interactive.
    mode='full'  : full CC_CAP rows, 300 iters, 5-fold cross-conformal (the seed-grade
                   path) — use occasionally to re-anchor calibration.
    """
    seed = get_seed()
    sc = _stable_cache()
    X, y0, lonlat = sc["X"], sc["y"], sc["lonlat"]
    cls_to_col = {int(c): j for j, c in enumerate(seed.classes)}
    labels = load_labels()
    y, weights = _fold_user_labels(y0, lonlat, labels, seed.classes, extra_weight)

    n_rows, iters, cc_k = ((15_000, 150, 1) if mode == "fast"
                           else (mc.CC_CAP, 300, mc.K_INNER))
    rng = np.random.default_rng(mc.SEED)
    n = len(X)
    if n > n_rows:                              # keep ALL user-labelled rows, fill rest
        forced = np.flatnonzero(weights > 1.0)
        pool = np.flatnonzero(weights == 1.0)
        take = max(n_rows - len(forced), 0)
        idx = np.concatenate([forced, rng.choice(pool, size=min(take, len(pool)),
                                                  replace=False)])
    else:
        idx = np.arange(n)
    Xs = X[idx]
    yc = np.array([cls_to_col[int(c)] for c in y[idx]])
    ws = weights[idx]

    model = mc.make_model(iterations=iters, n_classes=seed.n_classes)
    model.fit(Xs, yc, sample_weight=ws)

    if cc_k >= 2:
        pooled = mc.cross_conformal_aps(Xs, yc, cc_k, mc.SEED, seed.n_classes)
        tau = mc.conformal_quantile(pooled, mc.ALPHA)
    else:
        # single 50/50 split calibration (1 extra fit) — fast, τ is stable to it.
        from sklearn.model_selection import StratifiedKFold
        it, iv = next(StratifiedKFold(2, shuffle=True, random_state=mc.SEED).split(Xs, yc))
        cb = mc.make_model(iterations=iters, n_classes=seed.n_classes)
        cb.fit(Xs[it], yc[it])
        pv = cb.predict_proba(Xs[iv]).astype(np.float64)
        u = np.random.default_rng(mc.SEED).uniform(size=len(iv))
        tau = mc.conformal_quantile(mc.aps_cal_scores(pv, yc[iv], u), mc.ALPHA)

    new = mc.SeedModel(model=model, tau=tau, classes=seed.classes,
                       class_names=seed.class_names, alpha=mc.ALPHA)
    holder = _seed_holder()
    holder["model"] = new
    holder["version"] += 1
    build_predictions(new)                      # refresh predictions.parquet
    load_predictions.clear()                    # bust the data cache
    return holder["version"], tau, len(labels)


# ==============================================================================
# Map
# ==============================================================================
def build_map(points: pd.DataFrame, show_class_raster: bool, show_unc_raster: bool,
              raster_opacity: float, raster_stem: str | None):
    import folium
    from folium.raster_layers import ImageOverlay
    import rasterio
    from rasterio.warp import transform_bounds

    if len(points):
        center = [float(points.lat.median()), float(points.lon.median())]
    else:
        center = [59.10, 6.17]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    # dense raster overlays (PNG-encoded for the browser)
    if raster_stem and (show_class_raster or show_unc_raster):
        for kind, show in (("class", show_class_raster),
                           ("uncertainty", show_unc_raster)):
            tif = RASTER_DIR / f"{raster_stem}_{kind}_3857.tif"
            if show and tif.exists():
                img, bounds = _raster_to_rgba(tif, kind)
                ImageOverlay(image=img, bounds=bounds, opacity=raster_opacity,
                             name=f"{kind} raster", mercator_project=False).add_to(m)

    # sample points coloured by prediction, sized by uncertainty
    for r in points.itertuples():
        folium.CircleMarker(
            location=[float(r.lat), float(r.lon)],
            radius=3 + 1.4 * (int(r.set_size) - 1),
            color=CLASS_COLORS.get(int(r.pred), "#000000"),
            fill=True, fill_opacity=0.85, weight=1,
            tooltip=(f"pred={int(r.pred)} set_size={int(r.set_size)} "
                     f"grunnkart={int(r.grunnkart)}"),
            popup=f"{r.lon:.5f},{r.lat:.5f}",
        ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


@st.cache_data(show_spinner=False)
def _raster_to_rgba(tif_path: Path, kind: str):
    """Render a categorical/uncertainty GeoTIFF to an RGBA array + lon/lat bounds."""
    import rasterio
    from rasterio.warp import transform_bounds
    from matplotlib import cm

    with rasterio.open(tif_path) as ds:
        a = ds.read(1)
        b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    rgba = np.zeros((*a.shape, 4), dtype=np.uint8)
    valid = a > 0
    if kind == "class":
        from matplotlib.colors import to_rgba
        for code, hexc in CLASS_COLORS.items():
            mask = a == code
            if mask.any():
                rgba[mask] = (np.array(to_rgba(hexc)) * 255).astype(np.uint8)
    else:  # uncertainty: magma over set-size 1..max
        mx = max(int(a.max()), 1)
        norm = (a.astype(np.float64) / mx)
        col = (cm.magma(norm)[..., :4] * 255).astype(np.uint8)
        rgba[valid] = col[valid]
    rgba[~valid, 3] = 0  # transparent nodata
    bounds = [[b[1], b[0]], [b[3], b[2]]]  # [[south, west], [north, east]]
    return rgba, bounds


# ==============================================================================
# UI
# ==============================================================================
def main():
    seed = get_seed()
    version = _seed_holder()["version"]
    names = seed.class_names

    st.title("🛰️ Nature-type Active Learning")
    st.caption(f"Seed CatBoost (GPU) · {seed.n_classes} classes · APS α={seed.alpha} · "
               f"τ={seed.tau:.3f} · model v{version}")

    preds = load_predictions(version)
    labels = load_labels()
    labeled_keys = set(zip(labels.lon.round(8), labels.lat.round(8))) if len(labels) else set()

    # --- sidebar: controls -----------------------------------------------------
    with st.sidebar:
        st.header("Active-learning queue")
        only_unlabeled = st.checkbox("Hide already-labeled", value=True)
        only_disagree = st.checkbox("Only model≠grunnkart", value=False)
        queue_n = st.slider("Queue size", 10, 500, 100, step=10)
        max_points = st.slider("Max points on map", 100, 5000, 1000, step=100)

        st.divider()
        st.header("Map layers")
        rasters = list_rasters()
        raster_stem = None
        if rasters:
            stem = st.selectbox("Dense raster tile",
                                [r.name.replace("_class_3857.tif", "") for r in rasters])
            raster_stem = stem
        else:
            st.info("No dense raster yet — click *Regenerate map*.")
        show_class = st.checkbox("Class raster", value=True, disabled=not rasters)
        show_unc = st.checkbox("Uncertainty raster", value=False, disabled=not rasters)
        opacity = st.slider("Raster opacity", 0.0, 1.0, 0.6, 0.05)

        st.divider()
        if st.button("🔁 Retrain (fast)", use_container_width=True,
                     type="primary", disabled=len(labels) == 0,
                     help="~10s: 15k rows, 150 iters, single-split calibration"):
            with st.spinner(f"Fast retrain with {len(labels)} labels…"):
                v, tau, nlab = retrain_live(mode="fast")
            st.success(f"Retrained → v{v}, τ={tau:.3f}, {nlab} labels folded in.")
            st.rerun()
        if st.button("🎯 Full recalibrate", use_container_width=True,
                     disabled=len(labels) == 0,
                     help="Slower: full data, 300 iters, 5-fold cross-conformal τ"):
            with st.spinner("Full retrain + cross-conformal calibration…"):
                v, tau, nlab = retrain_live(mode="full")
            st.success(f"Full retrain → v{v}, τ={tau:.3f}.")
            st.rerun()
        if st.button("🗺️ Regenerate dense map", use_container_width=True):
            with st.spinner("Predicting dense tile…"):
                import subprocess
                r = subprocess.run(
                    ["python", str(Path(__file__).with_name("predict_raster.py"))],
                    capture_output=True, text=True)
            list_rasters.clear()
            (st.success if r.returncode == 0 else st.error)(
                r.stdout.splitlines()[-1] if r.stdout else "done")
            st.rerun()

    # --- build the queue --------------------------------------------------------
    q = preds.copy()
    if only_unlabeled and labeled_keys:
        keymask = ~q.apply(lambda r: (round(r.lon, 8), round(r.lat, 8)) in labeled_keys,
                           axis=1)
        q = q[keymask]
    if only_disagree:
        q = q[q.disagree]
    # acquisition = APS set size (desc), tie-break by lower max prob (more ambiguous)
    q = q.sort_values(["set_size", "p_max"], ascending=[False, True])
    queue = q.head(queue_n).reset_index(drop=True)

    col_map, col_panel = st.columns([3, 2], gap="medium")

    # --- map --------------------------------------------------------------------
    with col_map:
        from streamlit_folium import st_folium
        # show the most-uncertain points first, capped for responsiveness
        pts = q.head(max_points)
        m = build_map(pts, show_class, show_unc, opacity, raster_stem)
        st_folium(m, height=620, use_container_width=True, returned_objects=[])
        st.caption(f"Showing {len(pts):,} most-uncertain of {len(preds):,} locations · "
                   f"point size ∝ APS set size · colour = predicted class")

    # --- labeling panel ---------------------------------------------------------
    with col_panel:
        st.subheader("Label the queue")
        m1, m2, m3 = st.columns(3)
        m1.metric("Labeled", len(labels))
        m2.metric("Queue", len(queue))
        m3.metric("Mean set-size", f"{preds.set_size.mean():.2f}")

        if len(queue) == 0:
            st.info("Queue empty — adjust filters in the sidebar.")
        else:
            i = st.number_input("Queue index", 0, len(queue) - 1, 0)
            row = queue.iloc[int(i)]
            st.markdown(
                f"**Location** `{row.lon:.5f}, {row.lat:.5f}`  \n"
                f"**Model prediction:** {int(row.pred)} — *{names.get(str(int(row.pred)),'?')}*  "
                f"(p={row.p_max:.2f})  \n"
                f"**Grunnkart label:** {int(row.grunnkart)} — *{names.get(str(int(row.grunnkart)),'?')}*  \n"
                f"**APS set size:** {int(row.set_size)}  "
                f"{'⚠️ disagrees' if row.disagree else '✓ agrees'}")

            options = [int(c) for c in seed.classes]
            default = options.index(int(row.pred))
            choice = st.radio(
                "Assign class",
                options,
                index=default,
                format_func=lambda c: f"{c} — {names.get(str(c), '?')}",
                horizontal=False,
            )
            c1, c2 = st.columns(2)
            if c1.button("✅ Save label", use_container_width=True, type="primary"):
                save_label(float(row.lon), float(row.lat), int(choice))
                st.toast(f"Labeled {row.lon:.4f},{row.lat:.4f} → {choice}")
                st.rerun()
            if c2.button("⏭️ Accept grunnkart", use_container_width=True):
                save_label(float(row.lon), float(row.lat), int(row.grunnkart))
                st.toast("Accepted grunnkart label")
                st.rerun()

        with st.expander("Class legend"):
            for c in seed.classes:
                st.markdown(
                    f"<span style='display:inline-block;width:12px;height:12px;"
                    f"background:{CLASS_COLORS.get(int(c),'#000')};margin-right:6px;'></span>"
                    f"{int(c)} — {names.get(str(int(c)), '?')}",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
