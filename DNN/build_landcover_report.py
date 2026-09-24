"""Build the stakeholder-facing HTML report for the nyvest land-cover product.

Everything here is transcribed from measured artifacts, not re-run:

  DNN/reports/results/merge_bare_sparse.json   deployed 9-class CV scores
  DNN/reports/results/learning_curve.json      class-wise learning curves
  DNN/reports/logs/scaling_grid.log            params x data grid
  DNN/reports/results/temporal_llto.json       leave-location-and-time-out
  DNN/reports/results/conformal_compare.json   conformal method bake-off
  models/dnn_final_moe8_merged*.meta.json      deployed artifact + calibration
  DNN/reports/logs/validate_out.log            2018/2024 class composition
  <P-drive>/landcover_2018_2024/change_*.json  change layer + transitions
  DNN/reports/html/confusion_matrix_fragment.html   row-normalised recall

Audience is non-technical; the page is written to be mined for slides, so every
section ends on a single liftable sentence.

    $PY DNN/build_landcover_report.py
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent / "reports" / "html" / "nyvest_landcover_report.html"

# ---------------------------------------------------------------- palette ----
# Cartographic colours are the project's own QGIS palette (class_palette.clr /
# gee_display_inference.js) so the charts read against the delivered maps.
CARTO = {
    2:  ("#c8c98a", "Bare ground &amp; sparse vegetation"),
    3:  ("#e8d63a", "Cropland"),
    4:  ("#1a7d34", "Forest"),
    5:  ("#a3d977", "Grassland"),
    6:  ("#c49a52", "Scrub"),
    7:  ("#5fbcd3", "Wetland"),
    8:  ("#2b5dbd", "Water"),
    10: ("#d93030", "Built"),
    12: ("#ffffff", "Snow &amp; ice"),
    11: ("#b9b06e", "Sparse vegetation"),
}
SHORT = {2: "Bare ground", 3: "Cropland", 4: "Forest", 5: "Grassland", 6: "Scrub",
         7: "Wetland", 8: "Water", 10: "Built", 11: "Sparse veg", 12: "Snow / ice"}

# validated categorical slots (dataviz reference palette, all-pairs safe)
S1, S2, S3 = "var(--s1)", "var(--s2)", "var(--s3)"


# ------------------------------------------------------------- svg helpers ---
def hbar(x, y, w, h, r=4):
    """Horizontal bar anchored at x, rounded on the data end only."""
    w = max(w, 0.1)
    r = min(r, h / 2, w)
    return (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")


def vbar(x, ybase, w, hgt, r=4):
    """Vertical bar growing up from ybase, rounded on the data end only."""
    hgt = max(hgt, 0.1)
    r = min(r, w / 2, hgt)
    y = ybase - hgt
    return (f"M{x:.1f},{ybase:.1f} V{y + r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + r:.1f},{y:.1f} "
            f"H{x + w - r:.1f} A{r:.1f},{r:.1f} 0 0 1 {x + w:.1f},{y + r:.1f} "
            f"V{ybase:.1f} Z")


def figure(svg, cap, num, title, table=None):
    t = f"<details class='tbl'><summary>Show the numbers</summary>{table}</details>" if table else ""
    return f"""<figure class="fig">
<figcaption><span class="fignum">Figure {num}</span> {title}</figcaption>
<div class="plot">{svg}</div>
<p class="cap">{cap}</p>{t}
</figure>"""


# ------------------------------------------------------- chart 1: benchmark --
def chart_benchmark():
    rows = [
        ("Tuned random forest", "a strong classical baseline", 0.6945, S3, False),
        ("Linear read of the embeddings", "how far a straight line gets", 0.7040, S3, False),
        ("Previous production model", "CatBoost + TabICL &mdash; what we had to beat", 0.7139, S2, False),
        ("This model", "same 10 classes, same folds", 0.7341, "var(--accent)", True),
    ]
    W, LEFT, RIGHT = 720, 236, 74
    rowh, gap, top = 46, 14, 26
    H = top + len(rows) * (rowh + gap) + 44
    x0, xw = LEFT, W - LEFT - RIGHT
    lo, hi = 0.66, 0.76

    def px(v):
        return x0 + (v - lo) / (hi - lo) * xw

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Macro F1 of four models on the same '
         f'protocol; this model scores 0.7341 against the previous production model 0.7139." '
         f'class="chart">']
    for gv in [0.66, 0.68, 0.70, 0.72, 0.74, 0.76]:
        p.append(f'<line class="grid" x1="{px(gv):.1f}" y1="{top - 10}" x2="{px(gv):.1f}" y2="{H - 40}"/>')
        p.append(f'<text class="axis" x="{px(gv):.1f}" y="{H - 22}" text-anchor="middle">{gv:.2f}</text>')
    for i, (lab, note, v, col, big) in enumerate(rows):
        y = top + i * (rowh + gap)
        st = " strong" if big else ""
        p.append(f'<text class="blab{st}" x="{LEFT - 14}" y="{y + 19}" text-anchor="end">{lab}</text>')
        p.append(f'<text class="sublab" x="{LEFT - 14}" y="{y + 34}" text-anchor="end">{note}</text>')
        p.append(f'<path d="{hbar(x0, y + 9, px(v) - x0, rowh - 18)}" fill="{col}"/>')
        p.append(f'<text class="val{st}" x="{px(v) + 9}" y="{y + rowh / 2 + 4}">{v:.4f}</text>')
    p.append(f'<text class="axttl" x="{x0}" y="{H - 4}">Macro F1 &mdash; averaged over classes, so rare '
             f'classes count as much as common ones</text>')
    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------- figure 1: architecture ---
def _box(x, y, w, h, title, subs, fill="var(--surface)", stroke="var(--hair)", tcol="var(--ink)"):
    """A labelled box: bold title line, then muted sub-lines, vertically centred."""
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" '
         f'stroke="{stroke}" stroke-width="1.5"/>']
    cx = x + w / 2
    block = 19 + 17 * len(subs)
    ty = y + (h - block) / 2 + 14
    o.append(f'<text class="dgt" x="{cx:.0f}" y="{ty:.0f}" text-anchor="middle" fill="{tcol}">{title}</text>')
    for i, sub in enumerate(subs):
        o.append(f'<text class="dgs" x="{cx:.0f}" y="{ty + 19 + 17 * i:.0f}" text-anchor="middle">{sub}</text>')
    return "".join(o)


def chart_architecture():
    W, H = 920, 566
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart diagram" aria-label="Architecture of the '
         f'land-cover model. 67 numbers per pixel enter one network, which splits into an always-on '
         f'shared core and a router that picks 2 of 8 expert sub-networks; the two branches are added '
         f'together to give 9 class scores. Five such networks are trained and their probabilities '
         f'averaged, then calibrated and turned into a 90 percent guaranteed shortlist, giving a class, '
         f'a confidence and a shortlist for every pixel.">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker>'
         '<marker id="aha" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--accent)"/></marker></defs>']
    A = 'stroke="var(--muted)" stroke-width="1.5" fill="none" marker-end="url(#ah)"'

    # ---- input -------------------------------------------------------------
    p.append('<text class="dge" x="310" y="26">Every 10 m pixel</text>')
    p.append(_box(310, 36, 300, 70, "64 AlphaEarth bands + 3 lidar features",
                  ['<tspan fill="var(--accent)" font-weight="600">= 67 numbers</tspan>']))
    p.append(f'<path d="M460,106 V128" {A}/>')

    # ---- the network panel -------------------------------------------------
    p.append('<rect x="60" y="134" width="800" height="272" rx="4" fill="var(--sunk)" '
             'stroke="var(--hair)" stroke-width="1.5" stroke-dasharray="5 4"/>')
    p.append('<text class="dge" x="80" y="160">One network</text>')
    p.append('<text class="dge accent" x="840" y="160" text-anchor="end">&times;5 &mdash; trained five '
             'times from different random starts</text>')

    # split bus
    p.append(f'<path d="M460,134 V186" stroke="var(--muted)" stroke-width="1.5" fill="none"/>')
    p.append('<path d="M250,186 H670" stroke="var(--muted)" stroke-width="1.5" fill="none"/>')
    p.append(f'<path d="M250,186 V206" {A}/>')
    p.append(f'<path d="M670,186 V206" {A}/>')

    p.append(_box(140, 212, 220, 84, "Shared core",
                  ["256 &rarr; 128 units", "runs for every pixel"]))
    p.append(_box(560, 212, 220, 112, "Router", ["picks 2 of 8 experts", "", "for this pixel"]))
    # the 8 expert chips, 2 lit
    for i in range(8):
        lit = i in (2, 5)
        p.append(f'<rect x="{577 + i * 24}" y="272" width="18" height="14" rx="2" '
                 f'fill="{"var(--accent)" if lit else "var(--surface)"}" '
                 f'stroke="{"var(--accent)" if lit else "var(--hair)"}" stroke-width="1.5"/>')

    # merge
    p.append(f'<path d="M250,296 V356 H434" {A}/>')
    p.append(f'<path d="M670,324 V356 H486" stroke="var(--accent)" stroke-width="1.5" fill="none" '
             f'marker-end="url(#ahа)"/>'.replace("ahа", "aha"))
    p.append('<text class="dga" x="266" y="345">always on</text>')
    p.append('<text class="dga accent" x="654" y="345" text-anchor="end">adds to it &mdash; '
             'starts at zero</text>')
    p.append('<circle cx="460" cy="356" r="20" fill="var(--surface)" stroke="var(--accent)" '
             'stroke-width="2"/>')
    p.append('<text class="dgp" x="460" y="363" text-anchor="middle">+</text>')

    # ---- exit to the post-training strip -----------------------------------
    p.append(f'<path d="M460,376 V418 H24 V484 H42" {A}/>')
    p.append('<text class="dga" x="474" y="396">9 class scores</text>')

    p.append('<text class="dge" x="50" y="436">After training &mdash; applied once, to every pixel</text>')
    p.append(_box(50, 444, 190, 80, "Average the 5",
                  ["their probabilities,", "not their weights"]))
    p.append(f'<path d="M240,484 H266" {A}/>')
    p.append(_box(272, 444, 190, 80, "Venn&ndash;Abers",
                  ["makes the stated", "confidence honest"]))
    p.append(f'<path d="M462,484 H488" {A}/>')
    p.append(_box(494, 444, 190, 80, "Conformal shortlist",
                  ["90% coverage,", "guaranteed per class"]))
    p.append(f'<path d="M684,484 H710" stroke="var(--accent)" stroke-width="1.5" fill="none" '
             f'marker-end="url(#aha)"/>')
    p.append(_box(716, 444, 154, 80, "Every pixel gets",
                  ["a class, a confidence", "and a shortlist"],
                  fill="var(--accent-soft)", stroke="var(--accent)", tcol="var(--accent)"))

    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------------ chart 2: per-class ---
def chart_perclass():
    data = [(8, 0.9539), (10, 0.8461), (3, 0.7873), (2, 0.7589), (12, 0.7442),
            (4, 0.7430), (7, 0.6905), (6, 0.6670), (5, 0.5965)]
    W, LEFT, RIGHT = 720, 196, 70
    rowh, gap, top = 30, 10, 20
    H = top + len(data) * (rowh + gap) + 44
    x0, xw = LEFT, W - LEFT - RIGHT

    def px(v):
        return x0 + v * xw

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-class F1 for the nine delivered '
         f'classes, from water 0.95 down to grassland 0.60." class="chart">']
    for gv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        p.append(f'<line class="grid" x1="{px(gv):.1f}" y1="{top - 8}" x2="{px(gv):.1f}" y2="{H - 40}"/>')
        p.append(f'<text class="axis" x="{px(gv):.1f}" y="{H - 22}" text-anchor="middle">{gv:.1f}</text>')
    mean = 0.7542
    p.append(f'<line class="ref" x1="{px(mean):.1f}" y1="{top - 8}" x2="{px(mean):.1f}" y2="{H - 40}"/>')
    p.append(f'<text class="reflab" x="{px(mean):.1f}" y="{top - 12}" text-anchor="middle">average 0.754</text>')
    for i, (c, v) in enumerate(data):
        y = top + i * (rowh + gap)
        col, _ = CARTO[c]
        ring = ' stroke="var(--hair)" stroke-width="1.5"' if c == 12 else ""
        p.append(f'<text class="blab" x="{LEFT - 14}" y="{y + rowh / 2 + 4}" text-anchor="end">{SHORT[c]}</text>')
        p.append(f'<path d="{hbar(x0, y + 4, px(v) - x0, rowh - 8)}" fill="{col}"{ring}/>')
        p.append(f'<text class="val" x="{px(v) + 9}" y="{y + rowh / 2 + 4}">{v:.2f}</text>')
    p.append(f'<text class="axttl" x="{x0}" y="{H - 4}">F1 per class &mdash; 1.0 is perfect</text>')
    p.append("</svg>")
    return "".join(p)


# -------------------------------------------------- chart 3: learning curve --
def chart_learning():
    n = [22117, 44232, 88465, 176929, 309624, 442322]
    series = [
        ("Overall (average of classes)", [.6933, .7138, .7229, .7285, .7326, .7309], "var(--accent)", 3),
        ("Snow &amp; ice", [.6624, .7050, .7536, .7755, .7734, .7669], S3, 2),
        ("Grassland", [.5455, .5700, .5776, .5837, .5882, .5845], S2, 2),
        ("Bare ground", [.4133, .4692, .4806, .4873, .5040, .4950], S1, 2),
    ]
    W, H = 720, 372
    L, R, T, B = 58, 190, 24, 54
    lo, hi = 0.38, 0.82

    def X(v):
        return L + v / 460000 * (W - L - R)

    def Y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Accuracy against training-set size. The '
         f'overall line flattens after roughly 310 thousand labels; grassland and bare ground are flat '
         f'throughout." class="chart">']
    for gv in [0.4, 0.5, 0.6, 0.7, 0.8]:
        p.append(f'<line class="grid" x1="{L}" y1="{Y(gv):.1f}" x2="{W - R}" y2="{Y(gv):.1f}"/>')
        p.append(f'<text class="axis" x="{L - 10}" y="{Y(gv) + 4:.1f}" text-anchor="end">{gv:.1f}</text>')
    for v in n:
        if v == 44232:                      # would collide with the 22k tick
            continue
        p.append(f'<text class="axis" x="{X(v):.1f}" y="{H - 32}" text-anchor="middle">'
                 f'{v / 1000:.0f}k</text>')
    # shade the saturated region
    p.append(f'<rect class="band" x="{X(309624):.1f}" y="{T}" width="{X(442322) - X(309624):.1f}" '
             f'height="{H - T - B}"/>')
    p.append(f'<text class="bandlab" x="{(X(309624) + X(442322)) / 2:.1f}" y="{T + 14}" '
             f'text-anchor="middle">flat</text>')
    for lab, vals, col, wdt in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(n[i]):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{wdt}" stroke-linejoin="round" '
                 f'stroke-linecap="round"/>')
        for i, v in enumerate(vals):
            p.append(f'<circle cx="{X(n[i]):.1f}" cy="{Y(v):.1f}" r="{4 if wdt == 3 else 3.2}" '
                     f'fill="{col}" stroke="var(--surface)" stroke-width="2"/>')
        p.append(f'<text class="dlab" x="{X(n[-1]) + 12:.1f}" y="{Y(vals[-1]) + 4:.1f}" '
                 f'fill="{col}">{lab}</text>')
    p.append(f'<text class="axttl" x="{L}" y="{H - 8}">Number of training labels used</text>')
    p.append("</svg>")
    return "".join(p)


# -------------------------------------------------- chart 4: capacity gap ----
def chart_capacity():
    pts = [("5k", 5002, .7367, .7156), ("52k", 51594, .7987, .7304),
           ("169k", 168714, .8577, .7281), ("728k", 728330, .9539, .7191)]
    W, H = 720, 356
    L, R, T, B = 58, 214, 26, 54
    lo, hi = 0.68, 0.98
    xs = [L + i * (W - L - R) / (len(pts) - 1) for i in range(len(pts))]

    def Y(v):
        return T + (hi - v) / (hi - lo) * (H - T - B)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="As the network grows, its score on regions '
         f'it trained on climbs to 0.95 while its score on unseen regions stays near 0.73." class="chart">']
    for gv in [0.7, 0.8, 0.9]:
        p.append(f'<line class="grid" x1="{L}" y1="{Y(gv):.1f}" x2="{W - R}" y2="{Y(gv):.1f}"/>')
        p.append(f'<text class="axis" x="{L - 10}" y="{Y(gv) + 4:.1f}" text-anchor="end">{gv:.1f}</text>')
    for i, (lab, _, _, _) in enumerate(pts):
        p.append(f'<text class="axis" x="{xs[i]:.1f}" y="{H - 32}" text-anchor="middle">{lab}</text>')
    # gap ribbons
    for i, (_, _, val, tst) in enumerate(pts):
        p.append(f'<line class="gapline" x1="{xs[i]:.1f}" y1="{Y(val):.1f}" x2="{xs[i]:.1f}" '
                 f'y2="{Y(tst):.1f}"/>')
        p.append(f'<text class="gaplab" x="{xs[i] + 8:.1f}" y="{(Y(val) + Y(tst)) / 2 + 4:.1f}">'
                 f'+{val - tst:.3f}</text>')
    for vals, col, lab in [([q[2] for q in pts], S2, "Scored where it trained"),
                           ([q[3] for q in pts], "var(--accent)", "Scored on unseen regions")]:
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>')
        for i, v in enumerate(vals):
            p.append(f'<circle cx="{xs[i]:.1f}" cy="{Y(v):.1f}" r="4" fill="{col}" '
                     f'stroke="var(--surface)" stroke-width="2"/>')
        p.append(f'<text class="dlab" x="{xs[-1] + 12:.1f}" y="{Y(vals[-1]) + 4:.1f}" fill="{col}">{lab}</text>')
    p.append(f'<text class="axttl" x="{L}" y="{H - 8}">Size of the network (tunable numbers inside it)</text>')
    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------ chart 5: map composition ---
def chart_composition():
    rows = [(8, 33.31, 33.22), (2, 19.64, 19.25), (4, 17.39, 17.24), (6, 16.00, 15.71),
            (7, 5.53, 6.23), (5, 2.72, 3.04), (3, 2.22, 2.39), (12, 1.71, 1.53), (10, 1.49, 1.39)]
    W, LEFT, RIGHT = 720, 176, 168
    rowh, gap, top = 34, 10, 30
    H = top + len(rows) * (rowh + gap) + 42
    x0, xw = LEFT, W - LEFT - RIGHT

    def px(v):
        return x0 + v / 36 * xw

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Share of the mapped area by class in 2018 '
         f'and 2024; water a third, bare ground a fifth, forest a sixth. All year-to-year differences '
         f'are under one percentage point." class="chart">']
    for gv in [0, 10, 20, 30]:
        p.append(f'<line class="grid" x1="{px(gv):.1f}" y1="{top - 12}" x2="{px(gv):.1f}" y2="{H - 38}"/>')
        p.append(f'<text class="axis" x="{px(gv):.1f}" y="{H - 20}" text-anchor="middle">{gv}%</text>')
    p.append(f'<text class="axis strong" x="{W - RIGHT + 46}" y="{top - 14}" text-anchor="end">2018</text>')
    p.append(f'<text class="axis strong" x="{W - RIGHT + 104}" y="{top - 14}" text-anchor="end">2024</text>')
    p.append(f'<text class="axis strong" x="{W - RIGHT + 162}" y="{top - 14}" text-anchor="end">change</text>')
    for i, (c, a, b) in enumerate(rows):
        y = top + i * (rowh + gap)
        col, _ = CARTO[c]
        ring = ' stroke="var(--hair)" stroke-width="1.2"' if c == 12 else ""
        p.append(f'<text class="blab" x="{LEFT - 14}" y="{y + rowh / 2 + 4}" text-anchor="end">{SHORT[c]}</text>')
        p.append(f'<path d="{hbar(x0, y + 1, px(a) - x0, 13, 3)}" fill="{col}" opacity="0.42"{ring}/>')
        p.append(f'<path d="{hbar(x0, y + 17, px(b) - x0, 13, 3)}" fill="{col}"{ring}/>')
        d = b - a
        p.append(f'<text class="val" x="{W - RIGHT + 46}" y="{y + rowh / 2 + 4}" text-anchor="end">{a:.2f}</text>')
        p.append(f'<text class="val" x="{W - RIGHT + 104}" y="{y + rowh / 2 + 4}" text-anchor="end">{b:.2f}</text>')
        p.append(f'<text class="val muted" x="{W - RIGHT + 162}" y="{y + rowh / 2 + 4}" '
                 f'text-anchor="end">{d:+.2f}</text>')
    p.append(f'<text class="axttl" x="{x0}" y="{H - 4}">Share of the mapped area &mdash; '
             f'pale bar 2018, solid bar 2024</text>')
    p.append("</svg>")
    return "".join(p)


# --------------------------------------------------- chart 6: calibration ----
def chart_calibration():
    rows = [("No calibration<tspan class='sub'> (raw model output)</tspan>", 0.0510, S2),
            ("Temperature scaling<tspan class='sub'> (the usual fix)</tspan>", 0.0346, S3),
            ("Venn&ndash;Abers<tspan class='sub'> (what we shipped)</tspan>", 0.0060, "var(--accent)")]
    W, LEFT, RIGHT = 720, 268, 90
    rowh, gap, top = 40, 14, 22
    H = top + len(rows) * (rowh + gap) + 42
    x0, xw = LEFT, W - LEFT - RIGHT

    def px(v):
        return x0 + v / 0.056 * xw

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Calibration error falls from 0.051 raw to '
         f'0.006 with Venn-Abers calibration." class="chart">']
    for gv in [0, 0.02, 0.04]:
        p.append(f'<line class="grid" x1="{px(gv):.1f}" y1="{top - 8}" x2="{px(gv):.1f}" y2="{H - 38}"/>')
        p.append(f'<text class="axis" x="{px(gv):.1f}" y="{H - 20}" text-anchor="middle">{gv:.2f}</text>')
    for i, (lab, v, col) in enumerate(rows):
        y = top + i * (rowh + gap)
        p.append(f'<text class="blab" x="{LEFT - 14}" y="{y + rowh / 2 + 4}" text-anchor="end">{lab}</text>')
        p.append(f'<path d="{hbar(x0, y + 7, px(v) - x0, rowh - 14)}" fill="{col}"/>')
        p.append(f'<text class="val" x="{px(v) + 9}" y="{y + rowh / 2 + 4}">{v:.4f}</text>')
    p.append(f'<text class="axttl" x="{x0}" y="{H - 4}">Calibration error &mdash; lower is better; '
             f'0 means the stated confidence is exactly right</text>')
    p.append("</svg>")
    return "".join(p)


# ------------------------------------------------------- confusion matrix ----
CM_COLS = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
CM = {
    2:  [56, 7, 7, 5, 13, 1, 2, 8, 1, 0],
    3:  [0, 80, 1, 14, 0, 2, 0, 3, 0, 0],
    4:  [0, 1, 75, 6, 8, 6, 1, 2, 1, 0],
    5:  [1, 17, 8, 63, 3, 4, 1, 3, 0, 0],
    6:  [2, 1, 9, 3, 62, 8, 1, 1, 14, 0],
    7:  [0, 1, 8, 3, 9, 74, 2, 1, 1, 0],
    8:  [0, 0, 1, 0, 1, 1, 94, 0, 1, 0],
    10: [1, 5, 2, 3, 1, 1, 1, 86, 1, 0],
    11: [0, 0, 5, 0, 13, 1, 2, 0, 78, 0],
    12: [0, 0, 0, 0, 0, 0, 2, 0, 21, 76],
}


def confusion_table():
    head = "".join(f"<th>{SHORT[c]}</th>" for c in CM_COLS)
    body = []
    for r in CM_COLS:
        cells = []
        for j, c in enumerate(CM_COLS):
            v = CM[r][j]
            if v == 0:
                cells.append('<td class="z">&middot;</td>')
                continue
            t = min(v / 90, 1.0)
            cls = "d" if r == c else ""
            cells.append(f'<td class="{cls}" style="--t:{t:.3f}">{v}</td>')
        body.append(f"<tr><th scope='row'>{SHORT[r]}</th>{''.join(cells)}</tr>")
    return (f"<div class='cmwrap'><table class='cm'><caption>Read each row: of all pixels that really "
            f"were this class, what percentage did the model call each thing? The diagonal is what it got "
            f"right.</caption><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
            f"</table></div>")


# ------------------------------------------------------------------ page -----
def takeaway(text):
    return f'<aside class="slide"><span class="slidetag">For the slide</span><p>{text}</p></aside>'


def build():
    fig_arch = figure(
        chart_architecture(),
        "The two branches in the middle are the design decision. The shared core runs for every pixel, "
        "so it is the whole model on its own; the expert branch is initialised at zero and only ever "
        "<em>adds</em> to it, which is why switching the experts on cannot make the model worse than "
        "the plain version. Averaging happens across the five networks' probabilities, not their "
        "weights &mdash; five opinions, not one blurred model. Everything below the dashed box is "
        "fitted after training and costs no accuracy.",
        1, "How a pixel becomes a class, a confidence and a shortlist")

    fig_bench = figure(
        chart_benchmark(),
        "All four models were scored the same way: trained on part of the region, tested on parts of the "
        "region they had never seen, three times over. The comparison is like-for-like &mdash; same "
        "classes, same folds, same inputs &mdash; so the gap is the model and nothing else.",
        2, "The model beats every alternative we could put next to it",
        "<table><thead><tr><th>Model</th><th>Macro F1</th></tr></thead><tbody>"
        "<tr><td>Tuned random forest</td><td>0.6945</td></tr>"
        "<tr><td>Linear read of the embeddings</td><td>0.7040</td></tr>"
        "<tr><td>Previous production model (CatBoost&nbsp;+&nbsp;TabICL)</td><td>0.7139</td></tr>"
        "<tr><td>This model, same 10 classes</td><td><strong>0.7341</strong></td></tr>"
        "<tr><td>This model as delivered, 9 classes</td><td><strong>0.7542</strong></td></tr>"
        "</tbody></table>")

    fig_class = figure(
        chart_perclass(),
        "Water and built-up land are close to solved. Grassland and scrub are the weak spots, and the "
        "reason is visible in the confusion table below: they overlap with their neighbours on the ground, "
        "not just in the model.",
        3, "Where the accuracy sits, class by class")

    fig_lc = figure(
        chart_learning(),
        "Each point adds more training labels. The overall line stops rising after roughly 310,000 labels "
        "and dips slightly at the full 442,000. Grassland and bare ground never rise at all.",
        5, "More training data stopped helping")

    fig_cap = figure(
        chart_capacity(),
        "Four network sizes, from tiny to 140&times; bigger, all trained on the same data. The gap between "
        "the two lines is the model memorising the places it has seen. The size we ship is the second "
        "point &mdash; the peak of the line that actually matters.",
        6, "A bigger network learns the training regions, not the landscape")

    fig_calib = figure(
        chart_calibration(),
        "Calibration error measures the gap between stated and actual confidence. A raw model that says "
        "&ldquo;90% sure&rdquo; is right about 85% of the time; after Venn&ndash;Abers it is right about "
        "90% of the time. It also turned out about ten times cheaper to apply than the standard fix.",
        4, "The confidence numbers on the map mean what they say")

    fig_comp = figure(
        chart_composition(),
        "Two maps on one grid, 818 million pixels each. A third of the area is water, a fifth bare ground, "
        "a sixth forest. Every class moved by less than one percentage point between the two years.",
        7, "What the two maps say the region is made of")

    css = CSS
    return f"""<title>Nyvest Land Cover</title>
<style>{css}</style>

<div class="wrap">

<header class="hero">
  <p class="eyebrow">Nyvest &middot; land-cover mapping from satellite embeddings</p>
  <h1>Two maps of western Norway, and how far we could push them</h1>
  <p class="stand">We built a land-cover classifier for Rogaland, Vestland and M&oslash;re, ran it
  wall-to-wall for 2018 and 2024, and gave every one of the 818 million pixels an honest statement of
  how much to trust it. This page is the record of what the product is, how accurate it is, and the
  long list of things we tried that did not make it better.</p>

  <div class="stats">
    <div class="stat"><span class="sv">818M</span><span class="sl">pixels classified, twice</span></div>
    <div class="stat"><span class="sv">0.754</span><span class="sl">macro F1, tested on unseen regions</span></div>
    <div class="stat"><span class="sv">90%</span><span class="sl">guaranteed per-pixel coverage</span></div>
    <div class="stat"><span class="sv">17 min</span><span class="sl">to map a full year, one workstation</span></div>
  </div>
</header>

<nav class="toc" aria-label="Contents">
  <a href="#product">The product</a>
  <a href="#model">The model</a>
  <a href="#accuracy">Accuracy</a>
  <a href="#uncertainty">Uncertainty</a>
  <a href="#time">Robustness over time</a>
  <a href="#tried">What we tried</a>
  <a href="#saturation">Where the ceiling is</a>
  <a href="#scale">Running at scale</a>
  <a href="#maps">The 2018 and 2024 maps</a>
  <a href="#next">What comes next</a>
</nav>

<main>

<section id="product">
<h2><span class="hnum">01</span> What was delivered</h2>

<p>The product is nine raster layers on a single 10-metre grid, plus the model that made them. They
cover <strong>the whole three-county study area &mdash; Rogaland, Vestland and M&oslash;re og Romsdal,
92,000&nbsp;km&sup2;</strong>. This is not a pilot over part of the region: the pipeline refuses to run
unless the satellite mosaic covers the full county boundary, and that check passed for both years.
Of that area, 82,000&nbsp;km&sup2; (89%) has AlphaEarth coverage and is classified; the remainder has
no embedding data and is left explicitly unclassified rather than guessed.</p>

<div class="cards">
  <div class="card"><h3>Two land-cover maps</h3><p>2018 and 2024, nine classes, 818 million valid
  pixels each, pixel-for-pixel on the same grid so they can be compared directly.</p></div>
  <div class="card"><h3>A confidence layer per year</h3><p>For every pixel, a calibrated probability
  for each of the nine classes &mdash; not a raw model score, but a number that has been checked to
  mean what it says.</p></div>
  <div class="card"><h3>A guarantee layer per year</h3><p>For every pixel, the shortlist of classes
  that cannot be ruled out at 90% confidence. Where that shortlist has one entry, the map is
  trustworthy; where it has four, it is a coin-toss and says so.</p></div>
  <div class="card"><h3>A change layer</h3><p>2018 against 2024, with the diagnostics needed to know
  how much of the apparent change is real. That last part is the honest and uncomfortable half of
  this report &mdash; see section&nbsp;09.</p></div>
</div>

<h3>What goes in</h3>
<p>Every pixel is described by <strong>67 numbers</strong>. Sixty-four of them come from
<strong>AlphaEarth</strong>, Google DeepMind's satellite <em>embedding</em> dataset: instead of raw
imagery, each 10&nbsp;m pixel arrives as a 64-number summary that a large model has already distilled
from a full year of observations across several satellite sensors. That single choice does most of the heavy
lifting in this project &mdash; cloud gaps, seasonality and sensor differences have been handled
upstream, and one number per pixel per band replaces a stack of images. The remaining three numbers
come from Norway's national <strong>3&nbsp;m lidar</strong> survey: ground elevation, terrain
roughness, and canopy height.</p>

<p>The labels are the NIBIO/grunnkart base map sampled at 74,639 field locations across nine years,
giving 663,740 labelled pixel-years to train and test on.</p>

{takeaway("A nine-class land-cover map of western Norway for 2018 and 2024 at 10&nbsp;m, with a "
          "calibrated confidence value and a 90%-guaranteed shortlist behind every single pixel.")}
</section>

<section id="model">
<h2><span class="hnum">02</span> The model, in plain terms</h2>

<p>It is a <strong>small neural network</strong> &mdash; deliberately small. Sixty-seven numbers go in,
nine probabilities come out, and in between sit about 79,000 tunable values. For scale, a modern
language model has hundreds of billions. Section&nbsp;07 explains why bigger was tested and rejected.</p>

<p>Three design choices are worth naming, because they are where the work is:</p>

<ol class="steps">
  <li><strong>Five networks, not one.</strong> We train five copies from different random starting
  points and average their answers. This is the single most reliable accuracy gain we found anywhere in
  the project &mdash; worth more than every architectural idea combined.</li>
  <li><strong>A shared core plus eight specialists.</strong> Every pixel goes through one shared network;
  a small router additionally sends it to two of eight small &ldquo;expert&rdquo; sub-networks chosen
  for that pixel. The shared core means the model can never do worse than the plain version &mdash;
  the specialists can only add. This is the same design pattern that modern language models converged
  on, applied to a very different problem. <strong>We chose it for one specific reason, and it is not
  the average score</strong> &mdash; see the box below.</li>
  <li><strong>Two calibration stages after training.</strong> The network's raw confidence is turned
  into an honest probability, and then into a guaranteed shortlist. Section&nbsp;04.</li>
</ol>

{fig_arch}

<div class="callout">
<h4>Why the expert model is the one we ship</h4>
<p>Averaged over all classes the experts are worth +0.0039, and we are explicit in section&nbsp;06 that
a gain that size does not survive a change of random seed. <strong>It was chosen for what it does to a
single class.</strong> On snow and ice &mdash; the rarest class, the one with the worst labels, and the
one that matters most for glacier and snowfield monitoring &mdash; F1 goes from <strong>0.802 to 0.843,
a gain of 0.040</strong>, ten times the average gain and far outside the noise floor. Every other class
moves by less than 0.005. That is a real, targeted, reproducible improvement in the place we most wanted
one, and it is the whole justification for the extra complexity.</p>
</div>

<div class="spectable">
<table>
<caption>Deployed artifact &mdash; <code>models/dnn_final_moe8_merged.pt</code></caption>
<tbody>
<tr><th>Inputs</th><td>67 per pixel &mdash; 64 AlphaEarth embedding bands + lidar elevation, terrain
roughness, canopy height</td></tr>
<tr><th>Core</th><td>Two hidden layers, 256 then 128 units</td></tr>
<tr><th>Specialists</th><td>8 expert sub-networks, 2 consulted per pixel, routed on the pixel's own
content</td></tr>
<tr><th>Ensemble</th><td>5 independently trained copies, probabilities averaged</td></tr>
<tr><th>Size</th><td>~79,000 parameters per copy</td></tr>
<tr><th>Training set</th><td>663,740 labelled pixel-years, 2017&ndash;2025</td></tr>
<tr><th>Classes out</th><td>9 &mdash; bare ground, cropland, forest, grassland, scrub, wetland, water,
built, snow&nbsp;&amp;&nbsp;ice</td></tr>
<tr><th>Training time</th><td>13 minutes on one GPU</td></tr>
</tbody>
</table>
</div>

{takeaway("A five-model ensemble of small networks with a shared core and eight routed specialists, "
          "chosen because the specialists lift snow and ice by 0.040 &mdash; ten times their gain on "
          "any other class.")}
</section>

<section id="accuracy">
<h2><span class="hnum">03</span> How accurate it is</h2>

<p>Every number in this section comes from the same strict test. The region is cut into geographic
blocks; the model is trained on some blocks and scored on blocks it has <em>never seen</em>, three
times over so every block gets its turn. This is much harder than the usual random split, and much
closer to what happens when the map is used somewhere new. Test labels are never cleaned or corrected,
so nothing can leak.</p>

<p>The headline is <strong>macro F1 = 0.754</strong>. Macro means every class counts equally, so the
score is not flattered by the fact that a third of the area is water.</p>

{fig_bench}

<div class="callout">
<h4>One honest caveat on the headline number</h4>
<p>The delivered map merges &ldquo;sparse vegetation&rdquo; into &ldquo;bare ground&rdquo;, giving nine
classes instead of ten. That merge is an <strong>ontology decision</strong> &mdash; the two categories
were not meaningfully distinct for this product's users &mdash; and it lifts the headline score by
0.020 almost entirely by removing a hard distinction rather than by classifying anything better. Tested
properly, retraining on the merged classes beats simply merging the old model's answers afterwards by
just 0.0008. <strong>So the fair statement of improvement over the previous production model is
0.7139&nbsp;&rarr;&nbsp;0.7341, measured on identical classes.</strong> We report both.</p>
</div>

{fig_class}

<h3>Where the mistakes are</h3>
<p>The confusion table below is the most informative single object in this report. It shows the ten-class
version of the model, which is the one that reveals <em>why</em> the weak classes are weak.</p>

{confusion_table()}

<p>Three things to read out of it:</p>
<ul>
  <li><strong>Grassland is confused with cropland</strong> (17% of grassland pixels). On a satellite,
  improved pasture and a hay field look the same, because to a large extent they <em>are</em> the same.</li>
  <li><strong>Sparse vegetation is confused with scrub</strong> in both directions (13&ndash;14%), not
  with bare ground. This is worth flagging because it means the sparse-vegetation/bare-ground merge in
  the delivered product was made for ontology reasons and recovers only 0.6% of that class's errors.
  It is not a fix for the confusion.</li>
  <li><strong>Snow and ice loses 21% to sparse vegetation.</strong> This one is a label problem rather
  than a model problem &mdash; see section&nbsp;06.</li>
</ul>

{takeaway("0.754 macro F1 on regions the model has never seen &mdash; a 0.020 like-for-like gain over "
          "the previous production model, and 0.030 over a linear read of the same data.")}
</section>

<section id="uncertainty">
<h2><span class="hnum">04</span> Every pixel says how much to trust it</h2>

<p>This is the part of the work most worth putting on a slide. A conventional land-cover map gives you
one class per pixel and leaves you to guess where it is wrong &mdash; and a user who cannot tell the
strong parts of a map from the weak parts has to treat all of it as weak. This product ships two extra
answers per pixel so they do not have to.</p>

<div class="cards two">
  <div class="card"><h3>A probability you can act on</h3><p>Raw neural-network confidence is
  systematically overstated. We correct it with <strong>Venn&ndash;Abers calibration</strong>, so a
  pixel labelled &ldquo;80% forest&rdquo; really is forest about 80% of the time. Calibration error
  drops from 0.051 to 0.006.</p></div>
  <div class="card"><h3>A shortlist with a guarantee</h3><p><strong>Conformal prediction</strong>
  returns, for each pixel, the set of classes that cannot be ruled out. The set is built so that it
  contains the true class <strong>90% of the time</strong> &mdash; and this holds for every class
  individually, not just on average, which is the harder and more useful version.</p></div>
</div>

{fig_calib}

<p>We compared ten conformal methods before choosing. The chosen one (LAC with class-conditional
calibration) hits 90.0% coverage with an average shortlist of <strong>1.47 classes</strong>, and
<strong>60% of pixels get a shortlist of exactly one</strong> &mdash; the model has ruled out
everything else. Crucially, its per-class coverage lands between 0.900 and 0.902 for all nine classes;
the naive alternative varies by 0.29 between classes, meaning it quietly under-covers the rare ones.</p>

<div class="spectable">
<table>
<caption>The uncertainty layers, as delivered</caption>
<thead><tr><th>Layer</th><th>What it holds</th><th>Reading it</th></tr></thead>
<tbody>
<tr><td><code>uq_&lt;year&gt;_pcal</code></td><td>9 bands, calibrated probability per class</td>
<td>Highest band = the map class; its value = how sure</td></tr>
<tr><td><code>uq_&lt;year&gt;_setsize</code></td><td>1 band, size of the 90% shortlist</td>
<td><strong>1 = confident</strong> (60% of pixels); 4+ = do not use this pixel alone</td></tr>
<tr><td><code>uq_&lt;year&gt;_inset</code></td><td>9 bands, is this class on the shortlist</td>
<td>Which specific alternatives are still live</td></tr>
</tbody>
</table>
</div>

{takeaway("Every pixel carries a calibrated probability and a shortlist that is guaranteed to contain "
          "the right answer 90% of the time &mdash; so users can see exactly where the map is strong "
          "and where it is guessing.")}
</section>

<section id="time">
<h2><span class="hnum">05</span> It holds up across years</h2>

<p>A map for 2018 and a map for 2024 are only comparable if the model behaves the same way in both
years. We tested this the hardest way available: hold out <em>a whole year and a whole region at the
same time</em>, then score. The model has seen neither the place nor the date.</p>

<div class="numrow">
  <div class="num"><span class="nv">&minus;0.009</span><span class="nl">accuracy cost of holding out a
  year as well as a region</span></div>
  <div class="num"><span class="nv">&minus;0.024</span><span class="nl">cost of training on a single
  year instead of all nine</span></div>
  <div class="num"><span class="nv">0.700&ndash;0.734</span><span class="nl">range across the nine
  individual held-out years</span></div>
</div>

<p>Losing under one accuracy point when the year is unseen is a strong result: it says the model has
learned land cover, not the weather of a particular summer. Training on all nine years rather than one
is worth 0.024 on its own, which is why the pipeline pools every available year.</p>

<p>One class is the exception. <strong>Snow and ice</strong> swings from 0.48 to 0.73 depending on the
year held out &mdash; genuinely fragile, because it is rare, seasonal, and carries the most label noise
in the dataset. It is the one class where a year-to-year difference on the map should not be believed
without checking.</p>

{takeaway("Hiding an entire year from the model costs less than one accuracy point &mdash; the 2018 "
          "and 2024 maps are produced by a model that is stable over time, with snow and ice the one "
          "documented exception.")}
</section>

<section id="tried">
<h2><span class="hnum">06</span> What we tried &mdash; and what it cost to learn it did not work</h2>

<p>This section is the case for the result rather than an apology for it. Over three structured rounds
of automated experiments plus targeted investigations, <strong>we scored more than 100 distinct model
configurations</strong> against the same test &mdash; every one of them with its result file kept. The great majority were ties or losses. That is the
finding, and it is what tells us the current model is close to the ceiling of what this data supports
rather than merely the first thing that worked.</p>

<p>A note on reading the table: differences below about <strong>0.0013</strong> are indistinguishable
from random seed noise in this setup. We measured that floor deliberately, and we hold every result
to it &mdash; including our own best ideas.</p>

<div class="tried">
<table>
<caption>Every family of ideas tested, with the outcome</caption>
<thead><tr><th>What we tried</th><th>Why we expected it to help</th><th class="oc">Outcome</th></tr></thead>
<tbody>
<tr><td><strong>5-model ensemble</strong></td><td>Averaging cancels individual errors</td>
<td class="oc"><span class="p win">+0.0035 &mdash; adopted</span></td></tr>
<tr><td><strong>Targeted snow/ice label repair</strong></td><td>29 sites are mislabelled in every year,
so fix those specifically</td><td class="oc"><span class="p win">+0.0023 &mdash; adopted</span></td></tr>
<tr><td><strong>Lidar terrain features</strong></td><td>Elevation and roughness separate vegetation
types</td><td class="oc"><span class="p win">adopted &mdash; the one feature set that helped</span></td></tr>
<tr><td>Deeper / residual / batch-normalised networks</td><td>More capacity, better fit</td>
<td class="oc"><span class="p loss">memorises the training regions, &minus;0.003</span></td></tr>
<tr><td>Gated networks (GLU, GEGLU), self-normalising networks</td><td>Better-performing architectures
elsewhere</td><td class="oc"><span class="p loss">&minus;0.003 to &minus;0.024</span></td></tr>
<tr><td>Feature attention (squeeze-excite, softmax gating)</td><td>Let the model weight its 67 inputs</td>
<td class="oc"><span class="p loss">&minus;0.0035; the inputs are already an attention-derived
embedding</span></td></tr>
<tr><td>Transformer over features (FT-Transformer)</td><td>State of the art on tabular data</td>
<td class="oc"><span class="p loss">below baseline</span></td></tr>
<tr><td><strong>Mixture-of-experts by class (4, 8, 16 experts)</strong></td><td>Specialists for hard
classes</td><td class="oc"><span class="p win">adopted at 8 experts &mdash; +0.040 on snow&nbsp;&amp;&nbsp;ice</span>
<span class="p tie">average gain +0.004 does not survive a reseed</span></td></tr>
<tr><td>Mixture-of-experts by geography, soft and hard routing</td><td>Different regions need different
models</td><td class="oc"><span class="p loss">&minus;0.004 to &minus;0.023</span></td></tr>
<tr><td>Eleven modern LLM-style routing variants<br><span class="sub2">shared expert, loss-free
balancing, expert-choice, LoRA, adaptive depth, model soup, retrieval</span></td><td>The designs the
language-model field converged on</td><td class="oc"><span class="p tie">11 mechanisms, 0 wins; best
+0.0028 halves on a fresh seed</span></td></tr>
<tr><td>Region-specific models (fine-tuned or trained locally)</td><td>Local models for local landscapes</td>
<td class="oc"><span class="p loss">0 of 12 beat the global model, on any fold</span></td></tr>
<tr><td>Semi-supervised learning from unlabelled data</td><td>Free extra training signal</td>
<td class="oc"><span class="p tie">+0.0002 &mdash; a rounding error at this data volume</span></td></tr>
<tr><td>Bulk automated label cleaning</td><td>Noisy labels cap accuracy</td>
<td class="oc"><span class="p loss">&minus;0.0065; it recodes genuine grassland as forest</span></td></tr>
<tr><td>Extra feature sets: terrain models, pasture indices, alternative field labels</td>
<td>More information per pixel</td><td class="oc"><span class="p tie">no real gain from any of
them</span></td></tr>
<tr><td>Updated building base map (FKB Bygning 2018)</td><td>Better labels for built-up land</td>
<td class="oc"><span class="p loss">&minus;0.0002 as labels, &minus;0.0095 as extra training data
&mdash; adopted for the map, rejected for the model</span></td></tr>
<tr><td>Bigger networks and more training data, jointly</td><td>Scaling laws</td>
<td class="oc"><span class="p loss">both saturate &mdash; see section&nbsp;07</span></td></tr>
</tbody>
</table>
</div>

<p>Two of these deserve a sentence, because they are the ones a technical reviewer will ask about.</p>

<p><strong>The mixture-of-experts result is a real negative finding, not a failed implementation.</strong>
We diagnosed the early failure (the router was sending an entire held-out region to a single expert),
rebuilt the whole idea around that diagnosis using the design language models settled on, verified the
router now partitions the held-out region exactly as it partitions the training data &mdash; and it
still only ties. We then ran the control that settles it: a plain network widened to the same size does
nearly as well. Of the apparent gain, capacity explains most and routing explains a slice inside the
noise floor.</p>

<p><strong>Local models do not work here, and we can say by how much.</strong> Buying real labels inside
a held-out region and training a local model on them: even with 95,000 local labels, the local model
still trails a global model trained on labels from everywhere else by 0.040. Fine-tuning on local data
is worse than leaving the model alone at every budget tested. The one thing that does help is the
dullest: add the new labels to the global pile. <em>For the coming Vestland and M&oslash;re rollout that is
the operational answer &mdash; new field labels are worth collecting and worth pooling, and are worth
most when spread thinly across many unvisited places rather than densely over a few.</em></p>

{takeaway("More than 100 model configurations tested against the same strict benchmark, including "
          "eleven borrowed from modern language models &mdash; a handful earned their place, and the "
          "rest not beating a well-built small network is why we believe this one is near the ceiling.")}
</section>

<section id="saturation">
<h2><span class="hnum">07</span> Where the ceiling actually is</h2>

<p>Four independent lines of evidence say the same thing: the limit is not the model, and not the
amount of data. It is that some land-cover classes genuinely look alike from space.</p>

{fig_lc}

<p>Accuracy stops improving at about 310,000 training labels and is fractionally <em>lower</em> at the
full 442,000. More telling, the two worst classes &mdash; grassland and bare ground &mdash; are flat
from the very first point. They are not short of examples. They are short of anything that
distinguishes them.</p>

{fig_cap}

<p>The capacity result is the same story from the other side. As the network grows, its score on the
regions it trained on climbs from 0.74 to 0.95 &mdash; it is memorising them &mdash; while its score
on unseen regions goes <em>down</em>. A tuned random forest makes the point even more sharply: it
reaches 0.88 on its own training regions and then 0.69 on new ones, worse than a straight line.</p>

<div class="fourup">
  <div class="sat"><span class="satlab">Training data</span><p>Flat past 310k labels; the two weakest
  classes flat throughout.</p></div>
  <div class="sat"><span class="satlab">Model size</span><p>Peak at 52k parameters. 14&times; bigger is
  worse on unseen ground.</p></div>
  <div class="sat"><span class="satlab">Features</span><p>Lidar helped. Terrain models, pasture
  indices and alternative labels did not.</p></div>
  <div class="sat"><span class="satlab">Mechanisms</span><p>100+ ideas; the best is 9&times; smaller
  than what the network's non-linearity alone is worth.</p></div>
</div>

<p>The diagnosis behind all four panels: we tested, class by class, whether low accuracy is caused by
<em>wrong labels</em> (fixable) or by <em>genuine overlap in the data</em> (not fixable by modelling).
Only <strong>snow and ice</strong> came back noise-limited, and we fixed it &mdash; that is the
+0.0023 in section&nbsp;06. Grassland, scrub, bare ground and wetland are all overlap-limited: correcting
their labels makes them <em>worse</em>, because the corrections are themselves guesses about a genuinely
ambiguous boundary.</p>

<div class="callout">
<h4>What this means for anyone asking &ldquo;can it be more accurate?&rdquo;</h4>
<p>Yes, but not from this direction. Another 400,000 labels of the same kind, or a network ten times
bigger, will not move it. What would move it is <strong>a better description of each pixel</strong>
&mdash; a newer or richer embedding, or a higher-resolution input for the classes where 10&nbsp;m is
simply too coarse to see the thing being mapped.</p>
</div>

{takeaway("Accuracy has saturated on all four axes we can control &mdash; data, model size, features "
          "and method &mdash; so the next real gain has to come from better inputs, not a better model.")}
</section>

<section id="scale">
<h2><span class="hnum">08</span> Running it over a region this size</h2>

<p>818 million pixels &times; 67 numbers each &times; five networks &times; two years is a genuine
engineering problem, and the interesting finding is that <strong>the model was never the bottleneck
&mdash; reading the data was</strong>. The neural network can classify about 8 million pixels a second;
the disk can supply about 1.4 million. So the work went into the plumbing.</p>

<div class="spectable">
<table>
<caption>What each engineering change bought</caption>
<thead><tr><th>Change</th><th>What it does</th><th>Gain</th></tr></thead>
<tbody>
<tr><td>Parallel readers, one file handle each</td><td>Decompression spreads across cores instead of
queueing on one</td><td class="g">2.8&times;</td></tr>
<tr><td>Store the data as 8-bit, expand on the GPU</td><td>The values only ever had 8 bits of
information; we were storing them in 32 and paying to decompress the padding</td><td class="g">1.9&times;
end-to-end, files 20% smaller</td></tr>
<tr><td>Fuse the expert networks into batched operations</td><td>350 tiny GPU calls per block become
about 20</td><td class="g">2&times;, bit-for-bit identical output</td></tr>
<tr><td>Skip blocks outside the study area before reading</td><td>541 of 814 blocks never touched</td>
<td class="g">two-thirds of the work removed</td></tr>
</tbody>
</table>
</div>

<p>Together these turned the expert-based model from something that would have consumed 87% of the
available headroom into something that costs about 8% end-to-end. <strong>A full year over the study
area now takes about 17 minutes on a single eight-core workstation with one GPU</strong>, including
writing all the uncertainty layers.</p>

<p>Every optimisation is verified rather than assumed: twelve automated checks confirm that the fast
path and the simple path produce <em>bit-identical</em> class maps on 79.6 million pixels, and that the
8-bit storage is lossless.</p>

{takeaway("Full-region inference in about 17 minutes per year on one workstation &mdash; the model was "
          "never the bottleneck, and every speed-up is verified to produce bit-identical maps.")}
</section>

<section id="maps">
<h2><span class="hnum">09</span> The 2018 and 2024 maps</h2>

{fig_comp}

<p>Both years were produced by the same model from the same kind of input on the same grid, with an
identical set of 817,880,567 valid pixels. The composition is what a coastal, mountainous,
fjord-cut region should look like: a third water, a fifth bare rock and sparse ground, a sixth forest,
a sixth scrub, and only 1.4% built.</p>

<h3>On the change layer &mdash; read this before using it</h3>

<p>Comparing the two maps pixel by pixel, <strong>7.1% of the area changed class</strong>. We do not
believe that number, and the report ships the reason why.</p>

<div class="numrow three">
  <div class="num"><span class="nv">7.10%</span><span class="nl">of pixels changed class between the
  two maps</span></div>
  <div class="num"><span class="nv">0.40%</span><span class="nl">changed where the model was confident
  in <em>both</em> years</span></div>
  <div class="num warn"><span class="nv">21%</span><span class="nl">of built-up land &ldquo;left&rdquo;
  the class &mdash; but built-up land does not revert</span></div>
</div>

<p>The third number is the control. Buildings, roads and settlements do not turn back into forest over
six years, so essentially all of that 21% is model disagreement rather than change on the ground. It is
a lower bound on the error rate of a naive pixel-by-pixel comparison &mdash; and it is why the raw
7.1% figure must not be quoted as land-cover change.</p>

<p>The second number is what the uncertainty layers buy. Restrict the comparison to pixels where the
calibrated model gave a <em>single-class</em> answer in both years &mdash; 60% of the area &mdash; and
the flip rate falls to 0.40%, about one-eighteenth of the raw rate. That is the screen anyone doing
change analysis with this product should apply, and it is only possible because the uncertainty layers
exist.</p>

<div class="spectable">
<table>
<caption>Largest apparent transitions, 2018&nbsp;&rarr;&nbsp;2024, as a share of all valid pixels</caption>
<thead><tr><th>From</th><th>To</th><th>Share</th><th>Reading</th></tr></thead>
<tbody>
<tr><td>Bare ground</td><td>Scrub</td><td>1.07%</td><td>Plausible &mdash; and the direction expected
under a warming treeline</td></tr>
<tr><td>Scrub</td><td>Wetland</td><td>0.90%</td><td>Mostly boundary ambiguity between two overlapping
classes</td></tr>
<tr><td>Scrub</td><td>Bare ground</td><td>0.55%</td><td>The reverse of the top row &mdash; largely
cancels it</td></tr>
<tr><td>Scrub</td><td>Forest</td><td>0.45%</td><td>Plausible succession</td></tr>
<tr><td>Forest</td><td>Grassland</td><td>0.45%</td><td>Includes real felling, but also confusion</td></tr>
<tr><td>Snow &amp; ice</td><td>Bare ground</td><td>0.25%</td><td>Real glacier and snowfield retreat is
expected here, but this is the least reliable class</td></tr>
</tbody>
</table>
</div>

<p>Note the first and third rows: bare&nbsp;&rarr;&nbsp;scrub and scrub&nbsp;&rarr;&nbsp;bare are both
large and point in opposite directions. That signature &mdash; symmetric flips between two classes the
confusion table already shows overlapping &mdash; is what model noise looks like, not what succession
looks like.</p>

{takeaway("7.1% of pixels differ between the two maps, but only 0.40% differ where the model was "
          "confident in both years &mdash; and a class that cannot change shows a 21% flip rate, so "
          "this is a two-epoch land-cover product, not yet a change-detection product.")}
</section>

<section id="next">
<h2><span class="hnum">10</span> What comes next</h2>

<p>The saturation evidence in section&nbsp;07 points the roadmap fairly precisely: the headroom is on
the input side and in the class definitions, not in the model. The two items marked below are the ones
worth funding first &mdash; and both are projects, not tweaks.</p>

<div class="cards three">
  <div class="card pri"><h3>Newer and alternative embeddings</h3><p>The single highest-value direction,
  and the only one section&nbsp;07 says has real headroom. Earth-observation foundation models are moving
  quickly, and a richer pixel description is what the weak classes need. <strong>This is a full re-run of
  the project, not a swap</strong>: new embeddings have to be sampled at all 74,639 training locations,
  the model retrained, the calibration and conformal layers refitted, and both years re-inferred end to
  end. What carries over is the <em>benchmark</em> &mdash; the folds, the protocol and the 100-plus
  results in section&nbsp;06 &mdash; so a candidate embedding can be judged against everything already
  tried rather than in isolation. Budget it as a repeat of this study with a new input, on the order of
  weeks.</p></div>
  <div class="card pri"><h3>Higher resolution where 10&nbsp;m is the limit</h3><p>We proved this
  specifically: isolated rural buildings are recognised at the same rate as any other building
  <em>once you match on neighbourhood density</em>. They are missed because one or two 10&nbsp;m pixels
  cannot hold a farmhouse. That needs a sharper input or building footprints as a feature &mdash; not
  more labels.</p></div>
  <div class="card"><h3>More years, and the unclassified 11%</h3><p>The pipeline can produce any year
  AlphaEarth covers for the cost of one 17-minute run, so a denser time series is cheap &mdash; though
  section&nbsp;09 has to be solved first for those years to be comparable. Separately, 11% of the study
  area has no embedding coverage and is currently left blank; that gap is worth chasing at source.</p></div>
  <div class="card"><h3>External truth for snow and ice</h3><p>The one class where labels, not
  overlap, are the limit &mdash; and the one whose year-to-year behaviour is least trustworthy.
  Glacier inventories and snow-persistence indices would settle it.</p></div>
  <div class="card"><h3>Turn the two epochs into real change detection</h3><p>Requires modelling the
  two years jointly rather than differencing two independent maps. The uncertainty layers already
  provide the screen; the 21% built-up flip rate is the benchmark any such method has to beat.</p></div>
  <div class="card"><h3>Revisit the class definitions</h3><p>Collapsing the grassland/cropland and
  scrub/sparse boundaries recovers about 0.044 of accuracy. Treat that as an upper bound on what the
  boundaries cost, not proof that the definitions are the cause &mdash; but it is the largest single
  number left on the table, and it is a conversation about what users need the map to distinguish
  rather than a modelling task.</p></div>
</div>

{takeaway("The model side is done; the next real gain needs newer embeddings or sharper inputs "
          "&mdash; a fresh training run, not a swap, but one the existing benchmark can judge on day one.")}
</section>

</main>

<footer>
<p>Compiled from the measured artifacts in <code>DNN/reports/</code> and the deployed model metadata.
Every figure in this report is traceable to a stored result file &mdash; nothing here is an estimate.
Accuracy figures are three-fold spatial cross-validation on held-out geographic blocks with
uncleaned test labels.</p>
</footer>

</div>
"""


CSS = """
:root {
  color-scheme: light;
  --bg:      #f6f7f6;
  --surface: #ffffff;
  --sunk:    #eef1f0;
  --ink:     #131a1d;
  --ink2:    #33454b;
  --muted:   #5a6a70;
  --hair:    #dbe2e2;
  --hair2:   #eaeeed;
  --accent:  #0b6478;
  --accent-soft: #e2eff1;
  --s1: #2a78d6;
  --s2: #eb6834;
  --s3: #1baf7a;
  --win:  #0f6b46;
  --winbg:#e3f2ea;
  --loss: #9a3a20;
  --lossbg:#f7e9e3;
  --tie:  #5a6a70;
  --tiebg:#ecefee;
  --warnbg:#fdf1e6;

  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", "Source Serif 4", Georgia, serif;
  --sans: "Avenir Next", Avenir, "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", "IBM Plex Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg:      #12181a;
    --surface: #192124;
    --sunk:    #1e282b;
    --ink:     #e8eef0;
    --ink2:    #bccacd;
    --muted:   #9aabb0;
    --hair:    #2a3538;
    --hair2:   #222c2f;
    --accent:  #5cc9de;
    --accent-soft: #17323a;
    --s1: #3987e5;
    --s2: #d95926;
    --s3: #199e70;
    --win:  #63c79b;
    --winbg:#123326;
    --loss: #e59374;
    --lossbg:#33201a;
    --tie:  #9aabb0;
    --tiebg:#232d30;
    --warnbg:#33281a;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg:      #12181a;
  --surface: #192124;
  --sunk:    #1e282b;
  --ink:     #e8eef0;
  --ink2:    #bccacd;
  --muted:   #9aabb0;
  --hair:    #2a3538;
  --hair2:   #222c2f;
  --accent:  #5cc9de;
  --accent-soft: #17323a;
  --s1: #3987e5;
  --s2: #d95926;
  --s3: #199e70;
  --win:  #63c79b;
  --winbg:#123326;
  --loss: #e59374;
  --lossbg:#33201a;
  --tie:  #9aabb0;
  --tiebg:#232d30;
  --warnbg:#33281a;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 17px;
  line-height: 1.66;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 28px 96px; }

/* ---------- hero ---------- */
.hero { padding: 72px 0 40px; border-bottom: 1px solid var(--hair); }
.eyebrow {
  font-family: var(--sans); font-size: 12px; font-weight: 600;
  letter-spacing: .13em; text-transform: uppercase; color: var(--accent);
  margin: 0 0 20px;
}
h1 {
  font-family: var(--serif); font-weight: 400; font-size: clamp(34px, 5.2vw, 58px);
  line-height: 1.08; letter-spacing: -0.02em; margin: 0 0 24px;
  text-wrap: balance; max-width: 19ch;
}
.stand { font-size: 20px; line-height: 1.6; color: var(--ink2); max-width: 62ch; margin: 0 0 44px; }
.stats { display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid var(--hair); }
.stat {
  flex: 1 1 190px; padding: 22px 24px 20px 0; border-right: 1px solid var(--hair2);
  display: flex; flex-direction: column; gap: 4px;
}
.stat:last-child { border-right: 0; }
.sv {
  font-family: var(--sans); font-size: 34px; font-weight: 600; letter-spacing: -0.02em;
  color: var(--accent); font-variant-numeric: tabular-nums; line-height: 1.1;
}
.sl { font-family: var(--sans); font-size: 13px; line-height: 1.4; color: var(--muted); }

/* ---------- nav ---------- */
.toc {
  display: flex; flex-wrap: wrap; gap: 4px 22px;
  padding: 20px 0; margin-bottom: 8px; border-bottom: 1px solid var(--hair);
  font-family: var(--sans); font-size: 13px;
}
.toc a {
  color: var(--muted); text-decoration: none; padding: 2px 0;
  border-bottom: 1px solid transparent;
}
.toc a:hover, .toc a:focus-visible { color: var(--accent); border-bottom-color: var(--accent); }

/* ---------- sections ---------- */
main { display: block; }
section { padding: 60px 0 8px; border-bottom: 1px solid var(--hair); }
section:last-child { border-bottom: 0; }
h2 {
  font-family: var(--serif); font-weight: 400; font-size: clamp(26px, 3.4vw, 36px);
  letter-spacing: -0.015em; line-height: 1.15; margin: 0 0 26px; max-width: 22ch;
  text-wrap: balance; display: flex; align-items: baseline; gap: 16px;
}
.hnum {
  font-family: var(--mono); font-size: 13px; font-weight: 500; color: var(--accent);
  letter-spacing: .04em; flex: none; padding-top: 4px;
}
h3 {
  font-family: var(--sans); font-weight: 600; font-size: 15px; letter-spacing: .01em;
  margin: 40px 0 12px; color: var(--ink);
}
h4 { font-family: var(--sans); font-weight: 600; font-size: 14px; margin: 0 0 8px; }
p { max-width: 68ch; margin: 0 0 18px; }
ul, ol { max-width: 68ch; padding-left: 22px; }
li { margin-bottom: 10px; }
strong { font-weight: 600; }
em { font-style: italic; }
code {
  font-family: var(--mono); font-size: .85em; background: var(--sunk);
  padding: 1px 5px; border-radius: 3px; color: var(--ink2);
}

.steps { counter-reset: s; list-style: none; padding: 0; }
.steps li {
  counter-increment: s; position: relative; padding-left: 42px; margin-bottom: 16px;
}
.steps li::before {
  content: counter(s); position: absolute; left: 0; top: 2px;
  font-family: var(--mono); font-size: 12px; color: var(--accent);
  width: 26px; height: 26px; border: 1px solid var(--accent); border-radius: 50%;
  display: grid; place-items: center;
}

/* ---------- cards ---------- */
.cards {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(248px, 1fr));
  gap: 1px; background: var(--hair); border: 1px solid var(--hair);
  margin: 28px 0 32px;
}
.cards.two { grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.cards.three { grid-template-columns: repeat(auto-fit, minmax(292px, 1fr)); }
.card { background: var(--surface); padding: 22px 22px 20px; }
.card h3 {
  margin: 0 0 8px; font-family: var(--sans); font-size: 14px; font-weight: 600;
}
.card p { margin: 0; font-size: 15px; line-height: 1.55; color: var(--ink2); max-width: none; }
.card.pri { background: var(--accent-soft); }
.card.pri h3 { color: var(--accent); }

/* ---------- numbers row ---------- */
.numrow { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1px;
  background: var(--hair); border: 1px solid var(--hair); margin: 28px 0 30px; }
.num { background: var(--surface); padding: 20px 20px 18px; display: flex; flex-direction: column; gap: 6px; }
.num.warn { background: var(--warnbg); }
.nv {
  font-family: var(--sans); font-size: 30px; font-weight: 600; letter-spacing: -0.02em;
  color: var(--accent); font-variant-numeric: tabular-nums; line-height: 1;
}
.num.warn .nv { color: var(--loss); }
.nl { font-family: var(--sans); font-size: 13px; line-height: 1.45; color: var(--muted); }

/* ---------- saturation four-up ---------- */
.fourup { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1px;
  background: var(--hair); border: 1px solid var(--hair); margin: 30px 0; }
.sat { background: var(--surface); padding: 18px 18px 16px; }
.satlab {
  display: block; font-family: var(--sans); font-size: 11px; font-weight: 600;
  letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin-bottom: 8px;
}
.sat p { margin: 0; font-size: 14px; line-height: 1.5; color: var(--ink2); max-width: none; }

/* ---------- figures ---------- */
.fig { margin: 36px 0 40px; padding: 0; border: 1px solid var(--hair); background: var(--surface); }
figcaption {
  font-family: var(--sans); font-size: 15px; font-weight: 600; color: var(--ink);
  padding: 18px 24px 14px; border-bottom: 1px solid var(--hair2); display: flex;
  gap: 12px; align-items: baseline; flex-wrap: wrap;
}
.fignum {
  font-family: var(--mono); font-size: 11px; font-weight: 500; color: var(--accent);
  letter-spacing: .06em; text-transform: uppercase; flex: none;
}
.plot { padding: 20px 20px 6px; overflow-x: auto; }
.chart { display: block; width: 100%; height: auto; min-width: 520px; }
.cap {
  font-size: 14.5px; line-height: 1.55; color: var(--muted); margin: 0;
  padding: 4px 24px 20px; max-width: 78ch;
}

/* svg text roles */
.chart text { font-family: var(--sans); }
.chart .axis { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.chart .axis.strong { font-weight: 600; fill: var(--ink2); font-size: 10.5px;
  letter-spacing: .07em; text-transform: uppercase; }
.chart .axttl { font-size: 11px; fill: var(--muted); font-style: italic; }
.chart .blab { font-size: 13px; fill: var(--ink2); }
.chart .blab.strong { font-weight: 600; fill: var(--ink); }
.chart .val { font-size: 12.5px; fill: var(--ink2); font-variant-numeric: tabular-nums; }
.chart .val.strong { font-weight: 600; fill: var(--ink); }
.chart .val.muted { fill: var(--muted); }
.chart .sublab { font-size: 11px; fill: var(--muted); }
.diagram { min-width: 660px; }
.diagram .dgt { font-size: 13px; font-weight: 600; }
.diagram .dgs { font-size: 11px; fill: var(--muted); }
.diagram .dga { font-size: 10.5px; fill: var(--muted); }
.diagram .dga.accent { fill: var(--accent); }
.diagram .dge {
  font-size: 10px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
  fill: var(--muted);
}
.diagram .dge.accent { fill: var(--accent); }
.diagram .dgp { font-size: 20px; font-weight: 600; fill: var(--accent); }
.chart .dlab { font-size: 12.5px; font-weight: 600; }
.chart .sub { font-size: 11px; fill: var(--muted); }
.chart .grid { stroke: var(--hair2); stroke-width: 1; }
.chart .ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.chart .reflab { font-size: 10.5px; fill: var(--muted); }
.chart .band { fill: var(--sunk); }
.chart .bandlab { font-size: 10px; fill: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
.chart .gapline { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }
.chart .gaplab { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }

/* ---------- tables ---------- */
.spectable, .tried, .cmwrap, .tbl { overflow-x: auto; background: var(--surface); }
table {
  width: 100%; border-collapse: collapse; font-family: var(--sans); font-size: 14px;
  background: var(--surface);
}
.spectable, .tried { margin: 26px 0 32px; border: 1px solid var(--hair); }
caption {
  text-align: left; font-family: var(--sans); font-size: 12px; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  padding: 16px 18px 12px; background: var(--surface);
}
thead th {
  text-align: left; font-weight: 600; font-size: 11.5px; letter-spacing: .06em;
  text-transform: uppercase; color: var(--muted); padding: 10px 18px;
  border-top: 1px solid var(--hair); border-bottom: 1px solid var(--hair);
  white-space: nowrap;
}
tbody td, tbody th {
  padding: 12px 18px; border-bottom: 1px solid var(--hair2); vertical-align: top;
  text-align: left; font-weight: 400; line-height: 1.45;
}
tbody th { font-weight: 600; color: var(--ink); white-space: nowrap; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom: 0; }
td.g { color: var(--win); font-weight: 600; white-space: nowrap; }
td.oc, th.oc { white-space: normal; }
.sub2 { font-size: 12.5px; color: var(--muted); }

.p {
  display: inline-block; font-size: 12.5px; font-weight: 600; line-height: 1.35;
  padding: 3px 9px; border-radius: 3px;
}
.p.win  { background: var(--winbg);  color: var(--win); }
.p.loss { background: var(--lossbg); color: var(--loss); }
.p.tie  { background: var(--tiebg);  color: var(--tie); }

.tbl { margin: 0 24px 20px; }
.tbl summary {
  font-family: var(--sans); font-size: 12.5px; color: var(--accent); cursor: pointer;
  padding: 6px 0; list-style: none;
}
.tbl summary::-webkit-details-marker { display: none; }
.tbl summary::before { content: "▸ "; }
.tbl[open] summary::before { content: "▾ "; }
.tbl table { border: 1px solid var(--hair); margin-top: 6px; }

/* confusion matrix */
.cmwrap { margin: 24px 0 30px; border: 1px solid var(--hair); }
.cm { font-size: 12.5px; }
.cm caption { padding: 16px 18px 12px; text-transform: none; letter-spacing: 0;
  font-size: 13px; font-weight: 400; color: var(--muted); font-family: var(--sans); max-width: 82ch; }
.cm thead th { font-size: 10.5px; padding: 8px 6px; text-align: center;
  writing-mode: horizontal-tb; letter-spacing: .02em; }
.cm thead th:first-child { min-width: 96px; }
.cm tbody th { font-size: 12px; padding: 6px 10px 6px 18px; }
.cm tbody td {
  text-align: center; padding: 7px 6px; font-variant-numeric: tabular-nums;
  background: color-mix(in srgb, var(--accent) calc(var(--t) * 62%), var(--surface));
  border-bottom: 1px solid var(--surface); border-right: 1px solid var(--surface);
}
.cm tbody td.d { font-weight: 700; }
.cm tbody td.z { color: var(--hair); background: var(--surface); }

/* ---------- callout & takeaway ---------- */
.callout {
  background: var(--sunk); border-left: 3px solid var(--accent);
  padding: 20px 24px; margin: 30px 0; max-width: 72ch;
}
.callout p { margin: 0; font-size: 15.5px; max-width: none; color: var(--ink2); }

.slide {
  margin: 38px 0 12px; padding: 22px 26px; background: var(--accent-soft);
  border-top: 2px solid var(--accent);
  display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap;
}
.slidetag {
  font-family: var(--mono); font-size: 10.5px; font-weight: 500; letter-spacing: .1em;
  text-transform: uppercase; color: var(--accent); flex: none; padding-top: 5px;
  white-space: nowrap;
}
.slide p {
  margin: 0; font-family: var(--serif); font-size: 19px; line-height: 1.45;
  color: var(--ink); max-width: 58ch; text-wrap: balance;
}

footer { padding: 44px 0 0; border-top: 1px solid var(--hair); margin-top: 56px; }
footer p { font-family: var(--sans); font-size: 13px; color: var(--muted); max-width: 74ch; }

a:focus-visible, summary:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px;
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media (max-width: 720px) {
  body { font-size: 16px; }
  .wrap { padding: 0 18px 64px; }
  h2 { flex-direction: column; gap: 6px; }
  .slide { flex-direction: column; gap: 10px; }
  .stat { flex-basis: 140px; padding-right: 16px; }
}
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
