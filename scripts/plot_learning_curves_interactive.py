"""Render an interactive Plotly per-class learning-curves chart.

Reads `reports/learning_curves.csv` (produced by `scripts/learning_curves.py`)
and emits an HTML fragment with one subplot per class. The chart supports
scroll-zoom, pan, hover tooltips and per-trace legend toggling. Plotly's
JS bundle is included inline so the fragment can be embedded directly in
`reports/benchmark_cv_blocked_report.html` without external CDN calls.

Usage:
  ~/myprojects/recover/.venv/bin/python scripts/plot_learning_curves_interactive.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
REPORTS = _HERE.parent / "reports"
LC_CSV = REPORTS / "learning_curves.csv"
OUT_HTML = REPORTS / "_learning_curves_per_class_interactive.html"

CLASS_FREQ = {
    1: 352, 2: 600, 3: 7063, 4: 10945, 5: 4739, 6: 12771,
    7: 6538, 8: 10393, 9: 11341, 10: 5304, 11: 8914, 12: 159,
}


def main():
    df = pd.read_csv(LC_CSV)
    df = df.dropna(subset=["f1_macro"]).copy()
    class_cols = sorted(
        [c for c in df.columns if c.startswith("f1_class_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    n_classes = len(class_cols)
    models = sorted(df["model"].dropna().unique().tolist())

    # 4 cols × 3 rows for 12 classes
    ncols = 4
    nrows = int(np.ceil(n_classes / ncols))

    # Encoded indices 0..11 → raw labels 1..12
    subplot_titles = []
    for col in class_cols:
        enc = int(col.split("_")[-1])
        raw = enc + 1
        n = CLASS_FREQ.get(raw, "?")
        subplot_titles.append(f"class {raw} (n={n:,})" if isinstance(n, int)
                              else f"class {raw}")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.05, vertical_spacing=0.10,
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2"]
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(models)}

    for idx, col in enumerate(class_cols):
        r = idx // ncols + 1
        c = idx % ncols + 1
        for m in models:
            g = (df[df["model"] == m]
                 .groupby("n_train")[col]
                 .agg(["mean", "std"])
                 .reset_index()
                 .sort_values("n_train"))
            if g.empty:
                continue
            std = g["std"].fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=g["n_train"], y=g["mean"],
                    error_y=dict(type="data", array=std, visible=True,
                                 thickness=1, width=3),
                    mode="lines+markers",
                    name=m,
                    legendgroup=m,
                    showlegend=(idx == 0),
                    line=dict(color=model_color[m], width=1.5),
                    marker=dict(size=5),
                    hovertemplate=(
                        f"<b>{m}</b><br>"
                        "n_train=%{x:,}<br>"
                        "F1=%{y:.3f}<br>"
                        "± %{error_y.array:.3f}"
                        "<extra></extra>"
                    ),
                ),
                row=r, col=c,
            )

    # Axes: shared log x, fixed y range
    for axis_name in fig.layout:
        if axis_name.startswith("xaxis"):
            fig.layout[axis_name].update(type="log")
        if axis_name.startswith("yaxis"):
            fig.layout[axis_name].update(range=[-0.02, 1.02])

    fig.update_layout(
        title="Per-class learning curves (3-fold blocked CV) — interactive",
        height=240 * nrows + 100,
        margin=dict(l=50, r=20, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08,
                    xanchor="center", x=0.5),
        hovermode="closest",
        dragmode="zoom",
    )
    # Apply axis labels on the outer edges only
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text="n_train (log)", row=nrows, col=c)
    for r in range(1, nrows + 1):
        fig.update_yaxes(title_text="F1", row=r, col=1)

    config = dict(
        displaylogo=False,
        scrollZoom=True,
        responsive=True,
        modeBarButtonsToRemove=["lasso2d", "select2d"],
        toImageButtonOptions=dict(
            format="png", filename="learning_curves_per_class",
            scale=2,
        ),
    )

    OUT_HTML.parent.mkdir(exist_ok=True)
    # include_plotlyjs="cdn" keeps the fragment small; Plotly JS loads
    # from cdn.plot.ly when the page opens (requires internet).
    html_fragment = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config=config,
        div_id="lc-per-class-interactive",
    )
    OUT_HTML.write_text(html_fragment)
    print(f"Saved: {OUT_HTML}  ({OUT_HTML.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()