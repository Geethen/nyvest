"""Generate the research-loop dashboard from the experiment JSON records.

Reads DNN/research/results/*.json (written by exp_common.run_experiment) plus
the historical stage results, and emits a single self-contained HTML page.
Re-run after every batch; the Artifact tool redeploys it to the same URL.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LEGACY = HERE.parent / "reports" / "results"
OUT = HERE / "report.html"

BASELINE = 0.7318          # stage3 plain MLP + 5-seed ensemble
BEST = 0.7341              # stage8 to12_fix
TARGET_REF = 0.7139        # CatBoost+TabICL reference

# Historical results (already-run stages) so the page tells the whole story.
LEGACY_SPECS = [
    ("stage8 to12_fix relabel", "stage8_to12_fix.json", "label quality", "cls12-only surgical relabel"),
    ("stage3 MLP + 5-seed ens", "stage3_results.json", "baseline", "the model to beat"),
    ("class-MoE hard-specialist", "stage6_moe_moe_hard_results.json", "routing", "aux-supervised hard-class expert"),
    ("class-MoE (4 experts)", "stage6_moe_moe4_results.json", "routing", "learned gate over features"),
    ("stage5 GLU", "stage5_glu_results.json", "architecture", "gated linear unit"),
    ("stage9 feature attention (SE)", "stage9_attn_se.json", "attention", "squeeze-excite feature gate"),
    ("stage9 feature attention (softmax)", "stage9_attn_softmax.json", "attention", "softmax feature gate"),
    ("spatial-MoE soft e4", "stage7_spatial_soft_e4.json", "routing", "gate on (lon,lat)"),
    ("spatial-MoE soft e8", "stage7_spatial_soft_e8.json", "routing", "gate on (lon,lat), 8 experts"),
    ("stage5 residual-GLU", "stage5_residual_glu_results.json", "architecture", "deep residual + GLU"),
    ("spatial-MoE hard e4", "stage7_spatial_hard_e4.json", "routing", "KMeans region routing"),
    ("spatial-MoE hard e8", "stage7_spatial_hard_e8.json", "routing", "KMeans, 8 regions"),
    ("stage5 SNN", "stage5_snn_results.json", "architecture", "self-normalizing net"),
    ("stage2 ResMLP+BN+SWA", None, "architecture", "val 0.965 / test 0.71 — fold memorization"),
    ("stage1 plain MLP", None, "baseline", "no ensemble"),
]
LEGACY_FALLBACK = {"stage2 ResMLP+BN+SWA": 0.7106, "stage1 plain MLP": 0.7112}


def load_new():
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        if f.name.endswith("_probs.json"):
            continue
        d = json.loads(f.read_text())
        # skip the harness sanity run and any non-leaderboard record (e.g. the
        # ensemble diagnostic, which reports curves/oracle rather than an f1_mean)
        if d.get("name", "").startswith("sanity") or "f1_mean" not in d:
            continue
        rows.append({
            "name": d["name"], "notes": d.get("notes", ""),
            "f1": d["f1_mean"], "std": d.get("f1_std"),
            "per_fold": d.get("f1_per_fold"), "delta": d.get("delta_mean"),
            "delta_per_fold": d.get("delta_per_fold"),
            "verdict": d.get("verdict", "tie"), "new": True,
            "group": d.get("config", {}).get("idea", "architecture"),
            "runtime": d.get("runtime_s"), "per_class": d.get("f1_per_class"),
        })
    return rows


def load_legacy():
    rows = []
    for name, fn, group, note in LEGACY_SPECS:
        f1 = None
        if fn and (LEGACY / fn).exists():
            f1 = json.loads((LEGACY / fn).read_text())["f1_mean"]
        elif name in LEGACY_FALLBACK:
            f1 = LEGACY_FALLBACK[name]
        if f1 is None:
            continue
        d = round(f1 - BASELINE, 4)
        rows.append({
            "name": name, "notes": note, "f1": f1, "delta": d,
            "verdict": "WIN" if d > 0.003 else ("tie" if abs(d) <= 0.003 else "LOSS"),
            "new": False, "group": group, "per_fold": None, "delta_per_fold": None,
            "std": None, "runtime": None, "per_class": None,
        })
    return rows


def bar(delta, maxabs):
    """Diverging bar centered on the baseline. Returns (left%, width%, sign)."""
    if delta is None:
        return 0, 0, "tie"
    frac = max(-1.0, min(1.0, delta / maxabs)) if maxabs else 0
    half = abs(frac) * 50.0
    left = 50.0 - half if frac < 0 else 50.0
    return left, half, ("pos" if delta > 0 else "neg")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render():
    rows = load_new() + load_legacy()
    rows.sort(key=lambda r: r["f1"], reverse=True)
    maxabs = max([abs(r["delta"]) for r in rows if r["delta"] is not None] + [0.01])
    wins = [r for r in rows if r["verdict"] == "WIN" and r["new"]]
    n_new = sum(1 for r in rows if r["new"])
    best_row = rows[0]

    trs = []
    for r in rows:
        left, width, sign = bar(r["delta"], maxabs)
        d = r["delta"]
        dtxt = f"{d:+.4f}" if d is not None else "—"
        pf = ""
        if r["delta_per_fold"]:
            pf = " / ".join(f"{v:+.3f}" for v in r["delta_per_fold"])
        f1txt = f"{r['f1']:.4f}"
        std = f"±{r['std']:.3f}" if r["std"] else ""
        trs.append(f"""
      <tr class="{'is-new' if r['new'] else ''}">
        <td class="c-name">
          <span class="stripe v-{r['verdict'].lower()}"></span>
          <span class="nm">{esc(r['name'])}{'<span class="tag">new</span>' if r['new'] else ''}</span>
          <span class="note">{esc(r['notes'])}</span>
        </td>
        <td class="c-group"><span class="grp">{esc(r['group'])}</span></td>
        <td class="c-f1">{f1txt}<span class="std">{std}</span></td>
        <td class="c-bar">
          <div class="track"><span class="zero"></span>
            <span class="fill {sign}" style="left:{left:.2f}%;width:{width:.2f}%"></span>
          </div>
        </td>
        <td class="c-d {sign}">{dtxt}<span class="pf">{pf}</span></td>
        <td class="c-v"><span class="pill v-{r['verdict'].lower()}">{r['verdict']}</span></td>
      </tr>""")

    headline = (f"{len(wins)} experiment{'s' if len(wins)!=1 else ''} beat the baseline"
                if wins else "The ceiling is the data, not the model")
    sub = (f"Leader: <strong>{esc(wins[0]['name'])}</strong> at {wins[0]['f1']:.4f}."
           if wins else
           f"{n_new} architectures, training objectives and feature sets were tested against "
           f"a plain two-layer MLP. None beat it. The best model remains "
           f"<strong>{esc(best_row['name'])}</strong> at {best_row['f1']:.4f} — and the "
           f"pattern of failures says why: test accuracy is pinned near 0.73 however much "
           f"the model learns about the training regions.")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<title>DNN architecture research loop — nyvest</title>
<style>
  :root {{
    --bg:#F7F8F7; --panel:#FFFFFF; --ink:#141A18; --ink-2:#5A635E; --ink-3:#89918B;
    --line:#DFE3DF; --line-2:#EDEFEC;
    --accent:#2F6F5E; --pos:#3E7D3A; --neg:#A2472F; --tie:#8A8F86;
    --bar-pos:#3E7D3A26; --bar-neg:#A2472F26;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#0F1418; --panel:#161C20; --ink:#E8EDE9; --ink-2:#98A29B; --ink-3:#6C766F;
      --line:#242C30; --line-2:#1C2429;
      --accent:#5FA48C; --pos:#6FB35F; --neg:#D2795C; --tie:#7C857E;
      --bar-pos:#6FB35F2E; --bar-neg:#D2795C2E;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0F1418; --panel:#161C20; --ink:#E8EDE9; --ink-2:#98A29B; --ink-3:#6C766F;
    --line:#242C30; --line-2:#1C2429;
    --accent:#5FA48C; --pos:#6FB35F; --neg:#D2795C; --tie:#7C857E;
    --bar-pos:#6FB35F2E; --bar-neg:#D2795C2E;
  }}
  :root[data-theme="light"] {{
    --bg:#F7F8F7; --panel:#FFFFFF; --ink:#141A18; --ink-2:#5A635E; --ink-3:#89918B;
    --line:#DFE3DF; --line-2:#EDEFEC;
    --accent:#2F6F5E; --pos:#3E7D3A; --neg:#A2472F; --tie:#8A8F86;
    --bar-pos:#3E7D3A26; --bar-neg:#A2472F26;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased;
  }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:44px 24px 80px; }}
  .eyebrow {{
    font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--ink-3); font-weight:600; margin:0 0 10px;
  }}
  h1 {{ font-size:30px; line-height:1.18; letter-spacing:-.02em; margin:0 0 8px; text-wrap:balance; }}
  .sub {{ color:var(--ink-2); margin:0 0 30px; max-width:62ch; }}
  .sub strong {{ color:var(--ink); font-weight:600; }}

  .status {{
    display:flex; gap:1px; background:var(--line); border:1px solid var(--line);
    border-radius:10px; overflow:hidden; margin-bottom:14px; flex-wrap:wrap;
  }}
  .stat {{ background:var(--panel); padding:16px 20px; flex:1 1 160px; }}
  .stat .k {{
    font-size:10.5px; letter-spacing:.12em; text-transform:uppercase;
    color:var(--ink-3); font-weight:600; margin-bottom:6px;
  }}
  .stat .v {{
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:23px; font-variant-numeric:tabular-nums; letter-spacing:-.02em;
  }}
  .stat .v.accent {{ color:var(--accent); }}
  .stat .cap {{ font-size:12px; color:var(--ink-3); margin-top:3px; }}

  .panel {{
    background:var(--panel); border:1px solid var(--line);
    border-radius:10px; overflow:hidden;
  }}
  .panel-h {{
    display:flex; justify-content:space-between; align-items:baseline; gap:12px;
    padding:14px 18px; border-bottom:1px solid var(--line); flex-wrap:wrap;
  }}
  .panel-h h2 {{ font-size:14px; margin:0; letter-spacing:-.01em; }}
  .panel-h .hint {{ font-size:12px; color:var(--ink-3); }}
  .scroll {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; min-width:820px; }}
  th {{
    font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--ink-3);
    font-weight:600; text-align:left; padding:10px 12px; border-bottom:1px solid var(--line);
    white-space:nowrap; background:var(--panel);
  }}
  td {{ padding:11px 12px; border-bottom:1px solid var(--line-2); vertical-align:middle; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.is-new td {{ background:color-mix(in srgb, var(--accent) 4%, transparent); }}
  .c-name {{ position:relative; padding-left:22px; min-width:280px; }}
  .stripe {{ position:absolute; left:8px; top:12px; bottom:12px; width:3px; border-radius:2px; }}
  .stripe.v-win {{ background:var(--pos); }}
  .stripe.v-loss {{ background:var(--neg); }}
  .stripe.v-tie {{ background:var(--tie); }}
  .nm {{ display:block; font-weight:550; letter-spacing:-.01em; }}
  .tag {{
    display:inline-block; margin-left:7px; font-size:9.5px; letter-spacing:.1em;
    text-transform:uppercase; color:var(--accent); border:1px solid var(--accent);
    border-radius:3px; padding:0 4px; vertical-align:1px; font-weight:600;
  }}
  .note {{ display:block; font-size:12px; color:var(--ink-3); margin-top:2px; }}
  .grp {{
    font-size:11px; color:var(--ink-2); border:1px solid var(--line);
    border-radius:20px; padding:2px 9px; white-space:nowrap;
  }}
  .c-f1, .c-d {{
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-variant-numeric:tabular-nums; white-space:nowrap;
  }}
  .c-f1 {{ font-size:14.5px; }}
  .std {{ color:var(--ink-3); font-size:11.5px; margin-left:5px; }}
  .c-bar {{ width:190px; min-width:150px; }}
  .track {{ position:relative; height:22px; background:var(--line-2); border-radius:3px; }}
  .zero {{ position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--ink-3); opacity:.5; }}
  .fill {{ position:absolute; top:3px; bottom:3px; border-radius:2px; }}
  .fill.pos {{ background:var(--bar-pos); border-right:2px solid var(--pos); }}
  .fill.neg {{ background:var(--bar-neg); border-left:2px solid var(--neg); }}
  .c-d {{ font-size:13.5px; }}
  .c-d.pos {{ color:var(--pos); }}
  .c-d.neg {{ color:var(--neg); }}
  .pf {{ display:block; font-size:10.5px; color:var(--ink-3); margin-top:2px; letter-spacing:-.02em; }}
  .pill {{
    font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; font-weight:600;
    border-radius:3px; padding:2px 7px; white-space:nowrap;
  }}
  .pill.v-win {{ background:var(--bar-pos); color:var(--pos); }}
  .pill.v-loss {{ background:var(--bar-neg); color:var(--neg); }}
  .pill.v-tie {{ background:var(--line-2); color:var(--tie); }}

  .finding {{ margin-top:34px; }}
  .note-cell {{ font-size:12.5px; color:var(--ink-2); }}
  .method {{ margin-top:34px; display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1px;
             background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  .method div {{ background:var(--panel); padding:16px 18px; }}
  .method h3 {{ font-size:12px; margin:0 0 6px; letter-spacing:-.01em; }}
  .method p {{ margin:0; font-size:12.5px; color:var(--ink-2); line-height:1.55; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.9em;
          background:var(--line-2); padding:1px 4px; border-radius:3px; }}
  footer {{ margin-top:28px; font-size:12px; color:var(--ink-3); }}
</style>

<div class="wrap">
  <p class="eyebrow">nyvest · land-cover DNN · architecture search</p>
  <h1>{headline}</h1>
  <p class="sub">{sub} Every run uses the identical protocol — 663,740 rows, 67 features,
     3-fold GroupKFold on <code>cell_id</code>, 5-seed probability ensemble, train-only cls12
     cleaning — so a delta reflects the architecture and nothing else.</p>

  <div class="status">
    <div class="stat"><div class="k">Model to beat</div>
      <div class="v accent">{BASELINE:.4f}</div><div class="cap">plain 2-layer MLP + 5 seeds</div></div>
    <div class="stat"><div class="k">Overall best</div>
      <div class="v">{BEST:.4f}</div><div class="cap">+ cls12 surgical relabel</div></div>
    <div class="stat"><div class="k">Prior reference</div>
      <div class="v">{TARGET_REF:.4f}</div><div class="cap">CatBoost → TabICL</div></div>
    <div class="stat"><div class="k">New experiments</div>
      <div class="v">{n_new}</div><div class="cap">{len(wins)} win · this loop</div></div>
  </div>

  <div class="panel">
    <div class="panel-h">
      <h2>Every architecture tried, ranked by macro-F1</h2>
      <span class="hint">bar = delta vs baseline · per-fold deltas beneath</span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr>
          <th>Experiment</th><th>Family</th><th>Macro-F1</th>
          <th>Δ vs baseline</th><th>Δ</th><th>Verdict</th>
        </tr></thead>
        <tbody>{''.join(trs)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="method">
    <div><h3>Why paired per-fold deltas</h3>
      <p>Fold spread (~0.02) dwarfs the effects we chase (~0.002), so comparing run means
         mostly compares fold noise. The folds are fixed and deterministic, so each run is
         scored against the baseline <em>fold by fold</em>.</p></div>
    <div><h3>What counts as a win</h3>
      <p>A result must improve <em>every</em> fold and clear +0.003 mean — roughly one
         seed-noise sigma. Anything inside ±0.003 is recorded as a tie, not a gain.</p></div>
    <div><h3>What prior work established</h3>
      <p>Capacity is not the lever: residual nets hit 0.965 validation while test stayed at
         0.71. The hard vegetation classes are confusion-bound — learning curves flatten by
         70% of the data.</p></div>
  </div>

  <div class="panel finding">
    <div class="panel-h">
      <h2>What this loop found: test accuracy is pinned, whatever validation does</h2>
      <span class="hint">validation spans 22 points · test spans 3</span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr><th>Experiment</th><th>Validation F1</th><th>Test F1</th>
          <th>Gap</th><th>What it did to region information</th></tr></thead>
        <tbody>
          <tr><td class="c-name"><span class="nm">feat_temporal</span></td>
            <td class="c-f1">0.942</td><td class="c-f1">0.731</td><td class="c-f1">0.211</td>
            <td class="note-cell">added per-location phenology</td></tr>
          <tr><td class="c-name"><span class="nm">feat_neighbor</span></td>
            <td class="c-f1">0.864</td><td class="c-f1">0.703</td><td class="c-f1">0.161</td>
            <td class="note-cell">added spatial context</td></tr>
          <tr><td class="c-name"><span class="nm">baseline MLP</span></td>
            <td class="c-f1">0.790</td><td class="c-f1">0.732</td><td class="c-f1">0.058</td>
            <td class="note-cell">—</td></tr>
          <tr><td class="c-name"><span class="nm">SAM</span></td>
            <td class="c-f1">0.785</td><td class="c-f1">0.731</td><td class="c-f1">0.054</td>
            <td class="note-cell">sought flat minima</td></tr>
          <tr><td class="c-name"><span class="nm">Group-DRO</span></td>
            <td class="c-f1">0.781</td><td class="c-f1">0.726</td><td class="c-f1">0.055</td>
            <td class="note-cell">optimized the worst region</td></tr>
          <tr><td class="c-name"><span class="nm">IRM</span></td>
            <td class="c-f1">0.718</td><td class="c-f1">0.708</td><td class="c-f1">0.010</td>
            <td class="note-cell">discarded region-varying features</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="panel finding">
    <div class="panel-h">
      <h2>Why ensembling was the only lever — and why it stops at five</h2>
      <span class="hint">fold 0 · 15 members · the mechanism behind the one thing that worked</span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr><th>Ensemble size</th><th>Macro-F1</th><th>Gain over previous</th></tr></thead>
        <tbody>
          <tr><td class="c-name"><span class="nm">1 member</span></td>
            <td class="c-f1">0.7324</td><td class="note-cell">—</td></tr>
          <tr><td class="c-name"><span class="nm">3 members</span></td>
            <td class="c-f1">0.7363</td><td class="note-cell">+0.0039</td></tr>
          <tr><td class="c-name"><span class="nm">5 members</span></td>
            <td class="c-f1">0.7364</td><td class="note-cell">+0.0001 — saturated here</td></tr>
          <tr><td class="c-name"><span class="nm">15 members</span></td>
            <td class="c-f1">0.7373</td><td class="note-cell">+0.0009 across ten more members</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="method">
    <div><h3>The oracle bound closes the question</h3>
      <p>On fold 0 the 15-member ensemble is 78.7% accurate. If an oracle picked the right
         member for every row it would reach 85.1% — <strong>+6.3 points of headroom that no
         routing scheme can reach</strong>. Members disagree on just 5.9% of rows, and their
         disagreement does not track correctness.</p></div>
    <div><h3>Diversity contributes nothing</h3>
      <p>Fifteen architecturally diverse members (MLP + SwiGLU + RMSNorm + SAM + EMA) score
         0.7322; fifteen identical-architecture seeds score 0.7321. Same size, same budget —
         a 0.0001 difference. Only member count ever mattered, and only up to five.</p></div>
    <div><h3>The ceiling is information, not modelling</h3>
      <p>Validation ranges from 0.72 to 0.94 across these runs; test never leaves
         0.70–0.73. Everything added gets absorbed into fitting the training regions and
         none of it crosses the fold boundary.</p></div>
    <div><h3>Both directions lose</h3>
      <p>IRM <em>removed</em> region-specific features and lost 0.024. Neighbour features
         <em>added</em> them and lost 0.029. Those features are not shortcuts — they are the
         signal, which is why held-out geography stays hard.</p></div>
    <div><h3>Why lidar was different</h3>
      <p>Elevation and canopy height describe what is physically present, and physics holds
         across Norway. Neighbour statistics describe what is nearby <em>here</em>. Only
         transferable physical measurements have ever moved this number.</p></div>
  </div>

  <footer>Generated {ts} · <code>DNN/research/build_report.py</code></footer>
</div>
"""


if __name__ == "__main__":
    OUT.write_text(render())
    print(f"wrote {OUT}")
