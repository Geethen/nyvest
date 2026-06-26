"""Generate HTML report combining all three round-4 Tier-1 experiments."""

from __future__ import annotations

import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[2] / "common_ground" / "reports" / "research"

REF_BEST   = 0.6881
REF_LABEL  = "tabicl_25k_n16_kvon (prev best)"
WEAK_CLS   = [2, 5, 7]
WEAK_NAMES = {2: "bare", 5: "grassland", 7: "wetland"}

EXPERIMENTS = [
    ("round4a — Biased support draw",
     REPORT_DIR / "schemeB_round4_biased_support_results.json",
     "Force minority classes 2 (bare) and 5 (grassland) to higher shares "
     "of the 25k support set via stratified-without-replacement draw (no oversampling)."),
    ("round4b — Confidence-gated pseudo-labels",
     REPORT_DIR / "schemeB_round4_conf_gate_results.json",
     "Filter Stage-1 CatBoost pseudo-labels with low top-1 confidence (< 0.6) "
     "before they enter the Stage-2 TabICL support."),
    ("round4c — TabICL as Stage-1",
     REPORT_DIR / "schemeB_round4_tabicl_stage1_results.json",
     "Replace CatBoost-500 Stage-1 with TabICL (kv_cache=False, n_est=8 or 16) "
     "to generate higher-quality pseudo-labels for unstable data."),
]


def delta_cell(val, ref=REF_BEST):
    d = val - ref
    colour = "#2e7d32" if d > 0.003 else ("#c62828" if d < -0.003 else "#555")
    star = " ★" if d > 0.003 else ""
    return f'<td style="color:{colour};font-weight:bold">{d:+.4f}{star}</td>'


def f1_cell(val, threshold=0.65):
    colour = "#c62828" if val < threshold else "#2e7d32"
    return f'<td style="color:{colour}">{val:.4f}</td>'


def render_experiment(title, path, description):
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    decoded = data["merged_classes"]
    n_cls   = data["n_classes"]

    # sort by f1_mean descending
    sorted_names = sorted(results.keys(),
                          key=lambda n: results[n]["f1_mean"] if results[n]["f1_mean"] == results[n]["f1_mean"] else -1,
                          reverse=True)

    # leaderboard table
    lb_rows = ""
    for name in sorted_names:
        s = results[name]
        f1m  = s["f1_mean"]
        f1s  = s["f1_std"]
        balm = s["bal_mean"]
        dc   = delta_cell(f1m)
        lb_rows += (f"<tr><td><code>{name}</code></td>"
                    f"<td>{f1m:.4f}</td><td>{f1s:.4f}</td>"
                    f"<td>{balm:.4f}</td>{dc}"
                    f"<td style='font-size:0.85em;color:#555'>{s['description']}</td></tr>\n")

    # per-class F1 table (weak classes + macro)
    pc_header = "".join(f"<th>cls {c}<br><small>{WEAK_NAMES[c]}</small></th>" for c in WEAK_CLS)
    pc_rows = ""
    for name in sorted_names:
        s  = results[name]
        pc = s["f1_per_class"]
        cells = "".join(f1_cell(pc.get(str(c), float("nan"))) for c in WEAK_CLS)
        pc_rows += (f"<tr><td><code>{name}</code></td>"
                    f"<td><b>{s['f1_mean']:.4f}</b></td>{cells}</tr>\n")

    # per-fold detail
    fold_rows = ""
    for name in sorted_names:
        for fd in results[name]["per_fold"]:
            f1m = fd["f1_macro"]
            kept    = fd.get("n_pseudo_kept", "—")
            dropped = fd.get("n_pseudo_dropped", "—")
            fold_rows += (f"<tr><td><code>{name}</code></td><td>{fd['fold']}</td>"
                          f"<td>{f1m:.4f}</td><td>{fd['bal_acc']:.4f}</td>"
                          f"<td>{fd['n_support']:,}</td>"
                          f"<td>{kept}</td><td>{dropped}</td></tr>\n")

    return f"""
<section>
  <h2>{title}</h2>
  <p class="desc">{description}</p>

  <h3>Leaderboard (3-fold macro-F1, reference = {REF_BEST})</h3>
  <table>
    <thead><tr>
      <th>Condition</th><th>F1 mean</th><th>F1 std</th>
      <th>Bal acc</th><th>Δ ref</th><th>Description</th>
    </tr></thead>
    <tbody>{lb_rows}</tbody>
  </table>

  <h3>Per-class F1 — weak classes</h3>
  <table>
    <thead><tr>
      <th>Condition</th><th>Macro-F1</th>{pc_header}
    </tr></thead>
    <tbody>{pc_rows}</tbody>
  </table>

  <h3>Per-fold detail</h3>
  <table>
    <thead><tr>
      <th>Condition</th><th>Fold</th><th>F1</th><th>Bal acc</th>
      <th>n support</th><th>pseudo kept</th><th>pseudo dropped</th>
    </tr></thead>
    <tbody>{fold_rows}</tbody>
  </table>
</section>
<hr>
"""


def run():
    sections = ""
    for title, path, desc in EXPERIMENTS:
        sections += render_experiment(title, path, desc)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scheme B Round 4 — Tier-1 Improvements</title>
<style>
  body  {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }}
  h1   {{ border-bottom: 2px solid #1565c0; padding-bottom: .3em; }}
  h2   {{ color: #1565c0; margin-top: 2em; }}
  h3   {{ color: #333; }}
  table{{ border-collapse: collapse; width: 100%; margin: 1em 0 2em; font-size: .9em; }}
  th,td{{ border: 1px solid #ddd; padding: .45em .7em; text-align: right; }}
  th   {{ background: #e3f2fd; text-align: center; }}
  td:first-child {{ text-align: left; }}
  th:first-child {{ text-align: left; }}
  code {{ background: #f5f5f5; padding: .1em .3em; border-radius: 3px; }}
  .desc{{ background: #fafafa; border-left: 4px solid #1565c0;
          padding: .6em 1em; margin: .5em 0 1.5em; color: #444; }}
  .summary-box {{ background:#e8f5e9; border:1px solid #a5d6a7;
                  border-radius:6px; padding:1em 1.5em; margin:1.5em 0; }}
  .summary-box h3 {{ color:#2e7d32; margin-top:0; }}
</style>
</head>
<body>
<h1>Scheme B Round 4 — Tier-1 Improvements</h1>
<p><b>Date:</b> 2026-05-21 &nbsp;|&nbsp;
   <b>Reference:</b> {REF_LABEL} = <b>{REF_BEST}</b> &nbsp;|&nbsp;
   <b>Protocol:</b> 3-fold KMeans spatial LLTO, seed=0, score on stable_test only</p>

<div class="summary-box">
  <h3>Executive Summary</h3>
  <ul>
    <li><b>Round 4a (biased support) — NEW BEST:</b>
        <code>tabicl_biased_cls5</code> reaches <b>F1 = 0.6927</b> (+0.0046 ★),
        class-5 (grassland) F1 improves from 0.403 → 0.471.
        Forcing class 5 to ~17% of the 25k support is the single most effective lever found so far.
        Combining cls2+cls5 bias (biased_cls25) also beats the reference at 0.6904.</li>
    <li><b>Round 4b (confidence gate) — neutral/negative:</b>
        Gating weak-class pseudo-labels hurts macro-F1 (−0.0125 for gate_weak).
        The improvement in label quality is outweighed by the loss of support volume,
        especially in fold 2 where unstable data is scarce.
        gate_all is essentially flat (−0.0006).</li>
    <li><b>Round 4c (TabICL Stage-1) — marginal positive:</b>
        TabICL n8 (kv_cache=False) as Stage-1 scores 0.6869 (+0.0015 vs CatBoost Stage-1 re-run),
        but is below the reference best. TabICL Stage-1 improves class-2 pseudo-labels
        (mean_conf 0.74 vs CatBoost 0.46) but the downstream macro-F1 gain is small.</li>
  </ul>
  <p><b>New best overall: <code>tabicl_biased_cls5</code> at F1 = 0.6927</b>
     (prev best 0.6881, +0.0046). Recommended next step: combine biased_cls5 with
     TabICL n8 Stage-1 pseudo-labels to see if the two improvements stack.</p>
</div>

{sections}

</body>
</html>"""

    out = REPORT_DIR / "schemeB_round4_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved → {out}")


if __name__ == "__main__":
    run()
