"""Publication figures for the nyvest methods section (Nature style: 7 pt sans, 180 mm).

Inputs (written to $WORK, default ./work):
  maps.npz        python prep_map_arrays.py work/maps.npz      (fig1, fig4)
  oof_merged.npz  ARCH=moe_shared python oof_merged.py work/oof_merged.npz   (fig3, ~15 min GPU)
  tier4.json      python tally_nature_loss.py work/tier4.json   (fig5)
Run: python make_methods_figures.py fig1 fig2 fig3 fig4 fig5   -> figures/*.pdf, *.png (600 dpi)
"""
import os
import sys, json
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box

SP = Path(os.environ.get("WORK", Path(__file__).parent / "work"))
OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
    "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
    "ytick.major.size": 2, "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, MUTED, HAIR = "#1a1a1a", "#6b6b6b", "#bdbdbd"
CODES = [2, 3, 4, 5, 6, 7, 8, 10, 12]
NAMES = ["Bare ground & sparse veg.", "Cropland", "Forest", "Grassland", "Scrub",
         "Wetland", "Water", "Built", "Snow & ice"]
# project cartographic palette (class_palette.clr); snow tinted so it reads on white
COLS = ["#c8c98a", "#e8d63a", "#1a7d34", "#a3d977", "#c49a52", "#5fbcd3", "#2b5dbd",
        "#d93030", "#dce8f0"]
FOLD = ["#2a78d6", "#eb6834", "#1baf7a"]


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, **kw)
    plt.close(fig)


def label(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right")


def class_img(a):
    lut = np.zeros(13, np.uint8)
    for i, c in enumerate(CODES):
        lut[c] = i + 1
    return lut[np.clip(a, 0, 12)]


CMAP = ListedColormap(["#ffffff"] + COLS)
NORM = BoundaryNorm(np.arange(-0.5, 10.5), CMAP.N)


def scalebar(ax, x0, y0, km, h):
    ax.add_patch(Rectangle((x0, y0), km * 1000, h, color=INK, lw=0, zorder=5))
    ax.text(x0 + km * 500, y0 + h * 2.2, f"{km:g} km", ha="center", va="bottom", fontsize=6,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))


def north(ax, x, y, s):
    ax.annotate("N", xy=(x, y), xytext=(x, y - s), ha="center", va="center", fontsize=7,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=INK, mutation_scale=6))


# ------------------------------------------------------------------ Fig 1 ---
def fig1():
    M = np.load(SP / "maps.npz")
    cty = gpd.read_file("/data/P-Prosjekter2/154001_nyvest/GIS/Boundaries/nyvest_fylker.shp").to_crs(32633)
    fr = np.load(Path("/home/geethen.singh/myprojects/nyvest/DNN/.cache") /
                 "frame_lidar_1to2-9to8-11to2_1780590068_1782470941.npz", allow_pickle=True)
    sys.path.insert(0, "/home/geethen.singh/myprojects/nyvest/DNN")
    import data_utils as du
    fold = np.empty(len(fr["y_enc"]), int)
    for k, tr, te in du.fold_indices(fr["y_enc"], fr["groups"]):
        fold[te] = k
    lon, lat = fr["lon"], fr["lat"]
    _, ui = np.unique(np.round(lon, 6) * 1e3 + np.round(lat, 6), return_index=True)
    pts = gpd.GeoSeries(gpd.points_from_xy(lon[ui], lat[ui]), crs=4326).to_crs(32633)
    px, py, pf = pts.x.values, pts.y.values, fold[ui]

    l, b, r, t = M["full_bounds"]
    minx, miny, maxx, maxy = cty.total_bounds
    pad = 8000
    ext = (minx - pad, maxx + pad, miny - pad, maxy + pad)

    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 150 * MM),
                            gridspec_kw={"wspace": 0.04})
    for ax in axs:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color(HAIR)

    ax = axs[0]
    cty.plot(ax=ax, color="#eeeeec", edgecolor=MUTED, lw=0.5, zorder=1)
    order = np.random.default_rng(0).permutation(len(px))
    ax.scatter(px[order], py[order], s=0.15, c=np.array(FOLD)[pf[order]], lw=0,
               rasterized=True, zorder=2)
    # 25 km sampling cells (EPSG:25832 grid) outlined for scale
    for _, rw in cty.iterrows():
        c = rw.geometry.representative_point()
        ax.text(c.x, c.y, rw["navn"], fontsize=6.5, ha="center", va="center", color=INK,
                zorder=4, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3, color=FOLD[k],
                                  label=f"Group {k + 1}") for k in range(3)],
              title="Test group\n(74,639 locations)", loc="lower right", frameon=True,
              facecolor="white", edgecolor="none", framealpha=0.9,
              title_fontsize=6, handletextpad=0.2, alignment="left")
    scalebar(ax, ext[0] + 15000, ext[2] + 15000, 100, 4000)
    north(ax, ext[0] + 30000, ext[3] - 20000, 40000)
    label(ax, "a", 0.0, 1.005)

    ax = axs[1]
    ax.imshow(class_img(M["full"]), cmap=CMAP, norm=NORM, extent=(l, r, b, t),
              interpolation="nearest", zorder=1)
    cty.boundary.plot(ax=ax, color=INK, lw=0.4, zorder=2)
    zb = M["zoom_bounds"]
    ax.add_patch(Rectangle((zb[0], zb[1]), zb[2] - zb[0], zb[3] - zb[1], fill=False,
                           ec=INK, lw=0.8, zorder=3))
    ax.text(zb[2] + 4000, zb[3], "Fig. 4", fontsize=6, va="top", zorder=3,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))
    ax.legend(handles=[Patch(fc=c, ec=HAIR, lw=0.4, label=n) for c, n in zip(COLS, NAMES)],
              loc="lower right", frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.9, handlelength=1.2, handleheight=1.0,
              title="Land cover 2024", title_fontsize=6, alignment="left")
    scalebar(ax, ext[0] + 15000, ext[2] + 15000, 100, 4000)
    label(ax, "b", 0.0, 1.005)
    save(fig, "fig1_study_area")


# ------------------------------------------------------------------ Fig 2 ---
def fig2():
    fig, ax = plt.subplots(figsize=(180 * MM, 78 * MM))
    ax.set_xlim(0, 180); ax.set_ylim(0, 78); ax.axis("off")

    def node(x, y, w, h, title, body, fc="#f4f4f2", ec=HAIR, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc=fc, ec=ec, lw=0.6))
        ax.text(x + 2, y + h - 2.2, title, fontsize=6.8, fontweight="bold", va="top", color=INK)
        ax.text(x + 2, y + h - 6.4, body, fontsize=5.8, va="top", color="#3a3a3a",
                linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=MUTED, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))

    cols = [2, 47, 92, 137]
    heads = ["1  Inputs", "2  Training data", "3  Model & confidence", "4  Maps & change"]
    for x, hd in zip(cols, heads):
        ax.text(x, 75, hd, fontsize=7, fontweight="bold", color=INK, va="top")
        ax.plot([x, x + 41], [70.5, 70.5], color=INK, lw=0.5)
    W, H = 41, 19
    ys = [48, 26, 4]
    node(cols[0], ys[0], W, H, "NIBIO grunnkart labels",
         "land-cover base map, 10 m\n13 codes grouped into\n9 map classes")
    node(cols[0], ys[1], W, H, "AlphaEarth embeddings",
         "64 values per pixel and year\nsummarising the year's\nsatellite record (10 m)")
    node(cols[0], ys[2], W, H, "Airborne lidar (3 m)",
         "elevation, terrain\nruggedness and canopy\nheight, averaged to 10 m")
    node(cols[1], ys[0], W, H, "Stable pixels only",
         "no sign of land-cover\nchange 2017–2020 (CCDC)")
    node(cols[1], ys[1], W, H, "Representative sampling",
         "per 25 km cell and class,\n100 contrasting pixels\n→ 74,639 locations")
    node(cols[1], ys[2], W, H, "Training table",
         "663,740 pixel-years\n(2017–2025); outdated\nsnow/ice labels corrected")
    node(cols[2], ys[0], W, H, "Neural network",
         "shared core + 8 specialist\nsub-networks; average of\n5 independently trained")
    node(cols[2], ys[1], W, H, "Tested on unseen regions",
         "3 groups of 25 km cells\nmacro-F1 0.755;\nproducer's & user's accuracy")
    node(cols[2], ys[2], W, H, "Per-pixel confidence",
         "calibrated probabilities\n+ shortlist of classes that\nholds the truth 90% of time")
    node(cols[3], ys[0], W, H, "Full maps, 2018 & 2024",
         "818 million pixels per year\non one shared 10 m grid")
    node(cols[3], ys[1], W, H, "Layers per year",
         "class · probability per class\n· shortlist size ·\nshortlist members")
    node(cols[3], ys[2], W, H, "Change 2018–2024",
         "raw change + screened\nnature loss, each with\nan error control")
    for y in ys:                                                   # inputs -> training data
        arrow(cols[0] + W, y + H / 2, cols[1], y + H / 2)
    for c in (1, 2, 3):                                            # down each column
        for i in range(2):
            arrow(cols[c] + W / 2, ys[i], cols[c] + W / 2, ys[i + 1] + H)

    def elbow(c):                                                  # bottom of col c -> top of col c+1
        xm = cols[c] + W + 2
        ax.plot([cols[c] + W, xm, xm], [ys[2] + H / 2, ys[2] + H / 2, ys[0] + H / 2],
                color=MUTED, lw=0.6, solid_joinstyle="miter")
        arrow(xm, ys[0] + H / 2, cols[c + 1], ys[0] + H / 2)
    elbow(1); elbow(2)
    save(fig, "fig2_workflow")


# ------------------------------------------------------------------ Fig 4 ---
def fig4():
    M = np.load(SP / "maps.npz")
    zb = M["zoom_bounds"]; ext = (zb[0], zb[2], zb[1], zb[3])
    cls = M["zoom_cls"]; valid = cls > 0
    pmax = np.where(valid, M["zoom_pmax"] / 60000.0, np.nan)
    st = M["zoom_set"].astype(float); st[~valid] = np.nan
    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 66 * MM), gridspec_kw={"wspace": 0.06})
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color(HAIR)
    axs[0].imshow(class_img(cls), cmap=CMAP, norm=NORM, extent=ext, interpolation="nearest")
    seq = plt.get_cmap("Blues")
    im1 = axs[1].imshow(pmax, cmap="viridis", vmin=0.2, vmax=1, extent=ext, interpolation="nearest")
    setc = ListedColormap(["#fde7c8", "#f6b26b", "#d9731f", "#8c3b0a"])
    im2 = axs[2].imshow(np.clip(st, 1, 4), cmap=setc, norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4),
                        extent=ext, interpolation="nearest")
    for ax, s, t in zip(axs, "abc", ["Land cover", "Calibrated probability of mapped class",
                                     "90% conformal prediction-set size"]):
        label(ax, s, 0.0, 1.01); ax.set_title(t, loc="left", x=0.04, pad=3)
    scalebar(axs[0], ext[0] + 800, ext[2] + 900, 4, 180)
    # Reserve an identical footer band under all three panels (legend / colorbar / colorbar)
    # via make_axes_locatable, rather than letting fig.colorbar() shrink only b and c -
    # that used to leave panel a's map taller and vertically offset from b and c.
    dividers = [make_axes_locatable(ax) for ax in axs]
    caxes = [d.append_axes("bottom", size="5%", pad=0.35) for d in dividers]
    caxes[0].axis("off")
    present = [i for i, c in enumerate(CODES) if (cls == c).any()]
    caxes[0].legend(handles=[Patch(fc=COLS[i], ec=HAIR, lw=0.4, label=NAMES[i]) for i in present],
                     loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                     handlelength=1.0, columnspacing=0.8, fontsize=5.5)
    cb = fig.colorbar(im1, cax=caxes[1], orientation="horizontal")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5)
    cb2 = fig.colorbar(im2, cax=caxes[2], orientation="horizontal", ticks=[1, 2, 3, 4])
    cb2.ax.set_xticklabels(["1", "2", "3", "≥4"]); cb2.outline.set_linewidth(0.4)
    cb2.ax.tick_params(labelsize=5.5)
    save(fig, "fig4_map_excerpt")




# ------------------------------------------------------------------ Fig 3 ---
def fig3():
    Z = np.load(SP / "oof_merged.npz")
    y, p = Z["y"], Z["P"].argmax(1)
    n = len(CODES)
    cm = np.zeros((n, n)); np.add.at(cm, (y, p), 1)
    rec = cm / cm.sum(1, keepdims=True) * 100
    L = json.load(open("/home/geethen.singh/myprojects/nyvest/DNN/reports/results/temporal_llto.json"))
    yrs = sorted(L["per_year"])
    m = np.array([L["per_year"][k]["macro_f1"] for k in yrs])
    s = np.array([L["per_year"][k]["macro_f1_std"] for k in yrs])

    fig = plt.figure(figsize=(180 * MM, 80 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(rec, cmap="Blues", vmin=0, vmax=100)
    short = ["Bare/sparse", "Cropland", "Forest", "Grassland", "Scrub", "Wetland", "Water",
             "Built", "Snow/ice"]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Reference class")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = rec[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > 55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Share of reference pixels (%)", fontsize=6); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=5.5)
    fig.text(0.02, 1.04, "a", fontsize=8, fontweight="bold", va="top")

    ax = fig.add_subplot(gs[1])
    x = np.arange(len(yrs))
    ref = L["spatial_only_ref"]["macro_f1"]
    ax.axhline(ref, color=MUTED, lw=0.6, ls=(0, (3, 2)), label=f"Spatial-only CV ({ref:.3f})")
    ax.errorbar(x, m, yerr=s, fmt="o", ms=3.2, color=FOLD[0], ecolor=FOLD[0], elinewidth=0.6,
                capsize=1.5, capthick=0.6, mec="white", mew=0.5)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    ax.set_xticks(x); ax.set_xticklabels(yrs, rotation=45, ha="right")
    ax.set_ylim(0.68, 0.75); ax.set_ylabel("Macro-F1 (10 classes)")
    ax.set_xlabel("Held-out year")
    ax.grid(axis="y", color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    fig.text(0.57, 0.97, "b", fontsize=8, fontweight="bold", va="top")
    save(fig, "fig3_accuracy")
    print("pooled macro-F1 check:", np.round(rec.diagonal(), 1))


# ------------------------------------------------------------------ Fig 5 ---
def fig5():
    """Nature-loss screening. Stage areas from nature_loss_2018_2024.json; per-class
    split of the final layer from tier4.json (tally of band 1 == 4 by band 2 x 2024 class)."""
    stages = ["All candidate\npixels", "After the three\nscreens", "After removing\npatches < 0.1 ha"]
    loss = np.array([18825, 8607, 6394]) / 1000
    rev = np.array([21041, 2611, 1202]) / 1000
    T = json.load(open(SP / "tier4.json"))
    src = [(4, "Forest"), (7, "Wetland"), (5, "Grassland"), (6, "Scrub"), (2, "Bare ground &\nsparse veg.")]
    built = np.array([T.get(f"{c}->10", 0) for c, _ in src]) / 100 / 1000
    crop = np.array([T.get(f"{c}->3", 0) for c, _ in src]) / 100 / 1000

    fig = plt.figure(figsize=(180 * MM, 70 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2 - 0.01, loss, w, color=FOLD[0], label="Nature → cropland or built (loss)")
    ax.bar(x + w / 2 + 0.01, rev, w, color="#9e9e9e", label="Cropland or built → nature (error control)")
    for i in range(3):
        ax.text(i, max(loss[i], rev[i]) + 0.5, f"ratio {loss[i] / rev[i]:.1f}", ha="center",
                va="bottom", fontsize=6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Area (thousand ha)"); ax.set_ylim(0, 24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

    ax = fig.add_subplot(gs[1])
    y = np.arange(len(src))[::-1]
    ax.barh(y, built, 0.6, color=COLS[7], label="to built")
    ax.barh(y, crop, 0.6, left=built + 0.012, color=COLS[1], label="to cropland")
    for yi, b, c in zip(y, built, crop):
        ax.text(b + c + 0.06, yi, f"{(b + c) * 1000:,.0f} ha", va="center", fontsize=6)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in src])
    ax.set_xlabel("Screened nature loss 2018–2024 (thousand ha)"); ax.set_xlim(0, 3.0)
    ax.legend(loc="lower right", frameon=False, handlelength=1.0)
    ax.grid(axis="x", color="#e6e6e6", lw=0.4); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, 1.04, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(0.53, 1.04, "b", fontsize=8, fontweight="bold", va="top")
    save(fig, "fig5_nature_loss")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        globals()[f]()
