"""Render every result JSON into the live progress page.

The page has to answer three questions at a glance:
  1. has anything actually beaten the deployed model?
  2. did it lift the WEAK classes without denting the strong ones?
  3. what has been ruled out, so the next idea is not a repeat?

So the leaderboard encodes the decision rule visually rather than describing it:
three fold squares (the all-folds-positive test) and a diverging tradeoff bar
(weak-class gain up, worst strong-class drop down). A trial that wins on the mean
but loses a fold, or that buys weak-class F1 out of the strong classes, LOOKS
wrong before you read a single number.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import ar_common as ac

OUT = Path(__file__).resolve().parent / "report.html"
WANDB_URL = "https://wandb.ai/singhg10/nyvest-dnn-autoresearch"

TIER_ORDER = ["reference", "locality", "decision-rule", "loss", "hierarchy",
              "classifier", "optimizer", "activation", "architecture", "transfer",
              "transductive", "combo"]
TIER_BLURB = {
    "reference": "Controls. The paired baseline every delta is measured against, "
                 "the no-class-weights arm the imbalance losses need, and the "
                 "classical floor — a linear probe and a tuned forest on the "
                 "same folds, which say how much of the score the deep net is "
                 "responsible for in the first place.",
    "locality": "One global model or many local ones? Every MoE that lost here "
                "before routed on POSITION, which a held-out region cannot be "
                "placed inside. These route on content, on terrain, or on nothing "
                "at all — and each keeps the global model as an always-on shared "
                "expert instead of replacing it with a partition.",
    "decision-rule": "Leave the network alone; change how its output is read. "
                     "Macro-F1's optimal rule is a per-class threshold, not argmax.",
    "loss": "Move the decision boundary during training — margins and priors "
            "aimed at the rare and weak classes.",
    "hierarchy": "Split the problem: an easy between-group decision and a hard "
                 "within-group one, each with its own parameters.",
    "classifier": "Keep the representation, retrain or rescale only the last layer.",
    "optimizer": "Same model, different search — a different implicit bias about "
                 "which solution gets found.",
    "activation": "Same everything, different nonlinearity.",
    "architecture": "Representations the flat MLP cannot express.",
    "transfer": "Move what one model knows into another.",
    "transductive": "Use the target region's own UNLABELLED pixels — always "
                    "available at deployment, since inference runs over rasters.",
    "combo": "Do two independent wins stack, or do they fix the same error twice?",
}

PRIOR = [
    ("capacity &amp; data", "Joint params x data grid: interior optimum at 256,128, "
     "monotone decline past it; val-test gap grows to +0.235 at 728k params."),
    ("regularisation", "18-cell dropout x weight-decay sweep; heavier reg shrinks "
     "the gap by dragging val down, test never rises."),
    ("spatial regularisation", "Spatial jitter and leave-cells-out early stopping "
     "both monotonically hurt; the gap is shift, not reclaimable slack."),
    ("mixture of experts", "9 variants (class, spatial-soft, spatial-hard, "
     "aux-supervised, Soft-MoE) all lose; oracle bound says no router can win."),
    ("ensemble diversity", "5 -&gt; 15 seeds buys +0.0003; 15 diverse architectures "
     "match 15 identical seeds to +0.0001."),
    ("feature attention", "SE-gate, softmax-gate and FT-Transformer self-attention "
     "all tie or lose — 64 AlphaEarth bands are already a learned embedding."),
    ("invariance objectives", "IRM -0.024 and neighbour features -0.029: removing "
     "AND adding region information both lose."),
    ("more labels", "NiN field labels are a wash; learning curves flat for the "
     "weak classes past 70% of the data."),
]


def findings_html():
    """Hand-written interpretation, kept in findings.json so the loop can
    regenerate the page without touching prose."""
    f = Path(__file__).resolve().parent / "findings.json"
    if not f.exists():
        return ""
    items = json.loads(f.read_text())
    cards = "".join(
        f'<article class="finding {esc(i.get("state", ""))}">'
        f'<h4>{esc(i["title"])}</h4><p>{esc(i["body"])}</p></article>'
        for i in items)
    return f"""<h2>What has come out of it</h2>
<p class="note">Interpretation, updated as trials land. A negative that identifies
WHERE the ceiling lives is worth more here than another tie.</p>
<div class="findings">{cards}</div>"""


def esc(s):
    return html.escape(str(s))


# The classical floor. Scored by the same harness and shown on the leaderboard,
# but never counted as trials: they are what the search is measured ABOVE, not
# candidates in it, and folding them into "N mechanisms scored" would overstate
# the search by two and drag the leaderboard's tail down by 0.04.
FLOOR = {"probe_linear", "rf_tuned"}


def trial_recs(recs):
    """Records that are actual scored trials — the one definition the tiles, the
    floor panel and the leaderboard all count with."""
    return [r for r in recs if "delta_mean" in r
            and r.get("verdict") != "upper-bound" and r["name"] not in FLOOR]


def load():
    recs = []
    for p in sorted(ac.RESULTS_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        # Diagnostics also live in results/ but are not trials — they have no
        # name, tier or paired delta, and their own panel renders them.
        if isinstance(rec, dict) and "name" in rec:
            recs.append(rec)
    return recs


def fold_squares(rec):
    if "delta_per_fold" not in rec:
        return '<span class="folds">' + "".join(
            f'<i class="sq nil" title="fold {i}: {v:.4f}"></i>'
            for i, v in enumerate(rec["f1_per_fold"])) + "</span>"
    return '<span class="folds">' + "".join(
        f'<i class="sq {"pos" if d > 0 else "neg"}" title="fold {i}: '
        f'{rec["f1_per_fold"][i]:.4f} (delta {d:+.4f})"></i>'
        for i, d in enumerate(rec["delta_per_fold"])) + "</span>"


def tradeoff_bar(rec):
    """Weak-class mean gain above the axis, worst strong-class change below.
    Scale is fixed at +/-0.03 so bars are comparable across the whole table."""
    if "weak_gain_mean" not in rec:
        return '<span class="tradeoff empty"></span>'
    scale = 0.03
    w = max(-1, min(1, rec["weak_gain_mean"] / scale))
    s = max(-1, min(1, rec["strong_drop_worst"] / scale))
    return (
        f'<span class="tradeoff" title="weak-class mean {rec["weak_gain_mean"]:+.4f}, '
        f'worst strong class {rec["strong_drop_worst"]:+.4f}">'
        f'<i class="bar weak {"up" if w >= 0 else "down"}" '
        f'style="--h:{abs(w)*100:.0f}%"></i>'
        f'<i class="axis"></i>'
        f'<i class="bar strong {"up" if s >= 0 else "down"}" '
        f'style="--h:{abs(s)*100:.0f}%"></i></span>')


def verdict_chip(rec):
    v = rec.get("verdict", "running")
    cls = {"WIN": "win", "tie": "tie", "LOSS": "loss",
           "baseline": "ref", "upper-bound": "bound",
           "split": "split"}.get(v, "tie")
    extra = ' <b class="nt">clean</b>' if rec.get("no_tradeoff") and v == "WIN" else ""
    return f'<span class="chip {cls}">{esc(v)}</span>{extra}'


def leaderboard(recs):
    scored = [r for r in recs if "delta_mean" in r]
    scored.sort(key=lambda r: (-r["delta_mean"]))
    base = next((r for r in recs if r["name"] == "baseline"), None)
    rows = []
    if base:
        rows.append(row_html(base, is_base=True))
    rows += [row_html(r) for r in scored]
    return "\n".join(rows)


def row_html(rec, is_base=False):
    d = rec.get("delta_mean")
    dtxt = "—" if d is None else f"{d:+.4f}"
    dcls = "" if d is None else ("gain" if d > 0.003 else
                                 "drop" if d < -0.003 else "flat")
    cd = rec.get("delta_vs_control_mean")
    ctrl = ("—" if cd is None else
            f'<span title="{esc(rec.get("control",""))}">{cd:+.4f}</span>')
    flags = []
    if rec.get("transductive"):
        flags.append('<span class="flag">transductive</span>')
    if rec.get("verdict") == "upper-bound":
        flags.append('<span class="flag bound">not a result</span>')
    return f"""<tr class="{'base' if is_base else ''}">
  <td class="nm"><a href="#{esc(rec['name'])}">{esc(rec['name'])}</a>
      {''.join(flags)}</td>
  <td class="tier">{esc(rec.get('tier', ''))}</td>
  <td class="num f1">{rec['f1_mean']:.4f}<em>&thinsp;&plusmn;{rec['f1_std']:.3f}</em></td>
  <td class="num delta {dcls}">{dtxt}</td>
  <td class="num ctrl">{ctrl}</td>
  <td>{fold_squares(rec)}</td>
  <td>{tradeoff_bar(rec)}</td>
  <td class="vd">{verdict_chip(rec)}</td>
</tr>"""


def class_panel(recs):
    """Per-class paired deltas: the classical floor first, then the best trials.

    Both column groups are the SAME measure on the SAME colour scale — paired
    ΔF1 against the baseline's own per-class scores. The floor columns saturate
    it and the trial columns barely register it, and that contrast is the point,
    so they are not given a second scale to make them comparable in size. Every
    cell is direct-labelled, so the exact value survives the saturation.
    """
    base = next((r for r in recs if r["name"] == "baseline"), None)
    if not base:
        return ""
    by = {r["name"]: r for r in recs}
    floor = [by[n] for n in ("probe_linear", "rf_tuned")
             if n in by and "delta_per_class" in by[n]]
    scored = [r for r in recs if "delta_per_class" in r
              and r.get("verdict") != "upper-bound" and r["name"] not in FLOOR]
    scored.sort(key=lambda r: -r["delta_mean"])
    top = scored[:6]
    classes = sorted(base["f1_per_class"], key=lambda c: base["f1_per_class"][c])
    names = {"2": "rock / sand", "3": "crop", "4": "forest", "5": "grassland",
             "6": "scrub", "7": "wetland", "8": "water", "10": "built",
             "11": "sparse veg", "12": "snow / ice"}
    short = {"probe_linear": "linear probe", "rf_tuned": "tuned RF"}

    def cell(r, c, extra=""):
        dv = r["delta_per_class"].get(c, 0.0)
        a = min(1.0, abs(dv) / 0.03)
        tone = "p" if dv > 0.0005 else "n" if dv < -0.0005 else "z"
        return (f'<td class="cell {tone} {extra}" style="--a:{a:.2f}" '
                f'title="{esc(r["name"])} class {c}: {dv:+.4f} '
                f'(F1 {r["f1_per_class"].get(c, 0):.3f} vs baseline '
                f'{base["f1_per_class"][c]:.3f})">{dv:+.3f}</td>')

    head = "".join(
        f'<th class="colhead flr{" edge" if i == 0 else ""}">'
        f'<span>{esc(short.get(r["name"], r["name"]))}</span></th>'
        for i, r in enumerate(floor))
    head += "".join(
        f'<th class="colhead{" edge" if i == 0 and floor else ""}">'
        f'<span>{esc(r["name"])}</span></th>' for i, r in enumerate(top))
    body = []
    for c in classes:
        f1 = base["f1_per_class"][c]
        kind = ("weak" if int(c) in ac.WEAK_CLASSES else
                "strong" if int(c) in ac.STRONG_CLASSES else "mid")
        cells = "".join(cell(r, c, "flr" + (" edge" if i == 0 else ""))
                        for i, r in enumerate(floor))
        cells += "".join(cell(r, c, "edge" if i == 0 and floor else "")
                         for i, r in enumerate(top))
        body.append(
            f'<tr><th class="cls {kind}"><b>{esc(names.get(c, c))}</b>'
            f'<span>{c}</span></th>'
            f'<td class="num base">{f1:.3f}</td>{cells}</tr>')
    grp = ""
    if floor:
        grp = (f'<tr class="grp"><td></td><td></td>'
               f'<td class="gl edge" colspan="{len(floor)}">what the deep net buys</td>'
               f'<td class="gl edge" colspan="{len(top)}">what the search moved</td></tr>')
    return f"""<div class="scroll"><table class="matrix">
<thead>{grp}<tr><th>class</th><th class="num">baseline</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table></div>"""


def log_cards(recs):
    by_tier = {}
    for r in recs:
        by_tier.setdefault(r.get("tier", "other"), []).append(r)
    out = []
    for tier in TIER_ORDER + [t for t in by_tier if t not in TIER_ORDER]:
        if tier not in by_tier:
            continue
        items = sorted(by_tier[tier], key=lambda r: -r.get("delta_mean", -9))
        cards = []
        for r in items:
            prov = (f'<p class="prov">{esc(r["provenance"])}</p>'
                    if r.get("provenance") else "")
            res = ("baseline" if "delta_mean" not in r else
                   f'{r["f1_mean"]:.4f} &nbsp; Δ {r["delta_mean"]:+.4f} &nbsp; '
                   f'folds {", ".join(f"{d:+.4f}" for d in r["delta_per_fold"])}')
            cards.append(f"""<article class="card" id="{esc(r['name'])}">
  <header><h4>{esc(r['name'])}</h4>{verdict_chip(r)}</header>
  <p class="idea">{esc(r.get('idea', ''))}</p>
  <p class="hyp"><span>Hypothesis</span>{esc(r.get('hypothesis', ''))}</p>
  {prov}
  <p class="res">{res}</p>
</article>""")
        out.append(f"""<section class="tier-group">
  <h3>{esc(tier)}</h3>
  <p class="blurb">{TIER_BLURB.get(tier, '')}</p>
  <div class="cards">{''.join(cards)}</div>
</section>""")
    return "\n".join(out)


def routing_panel(recs):
    """Where the router actually sent a region it had never seen.

    The single number that decides whether a gate is usable here is the share of
    a HELD-OUT fold landing on one expert, next to the same share on the training
    rows. Equal shares mean the partition travelled; a spike on the held-out side
    means the router extrapolated and the MoE quietly became one small expert.
    """
    rows = []
    for r in recs:
        info = [f for f in r.get("fold_info", []) if "te_route_max_share" in f]
        if not info:
            continue
        te = sum(f["te_route_max_share"] for f in info) / len(info)
        tr = sum(f.get("tr_route_max_share", 0) for f in info) / len(info)
        ent = sum(f.get("te_route_entropy_norm", 0) for f in info) / len(info)
        src = r.get("config", {}).get("gate_src", "—")
        rows.append((r["name"], src, tr, te, ent, r.get("delta_mean")))
    if not rows:
        return ""
    rows.sort(key=lambda x: x[3])
    body = "".join(
        f'<tr><td><a href="#{esc(n)}">{esc(n)}</a></td><td class="tier">{esc(s)}</td>'
        f'<td class="num">{tr:.2f}</td><td class="num">{te:.2f}</td>'
        f'<td class="num">{e:.2f}</td>'
        f'<td class="num">{d:+.4f}</td></tr>'
        for n, s, tr, te, e, d in rows if d is not None)
    return f"""<h2>Did the router survive the fold boundary?</h2>
<p class="note">Share of rows landing on the single busiest expert, on the training
rows and then on the held-out region, averaged over folds. stage7's hard spatial
gate put 84-95&#37; of a held-out fold on one expert — that is the failure this
column exists to detect. Normalised entropy is 1.00 for perfectly even use of all
experts and 0.00 for total collapse.</p>
<div class="scroll"><table>
<thead><tr><th>trial</th><th>gate reads</th><th class="num">train max share</th>
<th class="num">held-out max share</th><th class="num">entropy</th>
<th class="num">Δ paired</th></tr></thead>
<tbody>{body}</tbody></table></div>"""


def gap_panel(recs):
    """Inner val against held-out test, for the locality arms and the baseline.

    A mechanism that lifts val and leaves test alone HAS been learned; it just
    did not survive the fold boundary. That is a different result from one the
    optimiser never picked up, and on this model it is the result that keeps
    recurring — capacity, regularisation and now locality all land here."""
    keep = [r for r in recs
            if r.get("val_f1_mean") and (r.get("tier") in ("locality", "locality-v2")
                                         or r["name"] in ("baseline", "baseline_gval"))
            and r.get("verdict") != "upper-bound"]
    if len(keep) < 3:
        return ""
    keep.sort(key=lambda r: -(r["val_f1_mean"] - r["f1_mean"]))
    lo = min(min(r["val_f1_mean"], r["f1_mean"]) for r in keep) - 0.005
    hi = max(max(r["val_f1_mean"], r["f1_mean"]) for r in keep) + 0.005
    span = max(hi - lo, 1e-6)
    rows = []
    for r in keep:
        v, t = r["val_f1_mean"], r["f1_mean"]
        x1, x2 = (v - lo) / span * 100, (t - lo) / span * 100
        base = r["name"] in ("baseline", "baseline_gval")
        rows.append(
            f'<tr class="{"base" if base else ""}">'
            f'<td class="nm"><a href="#{esc(r["name"])}">{esc(r["name"])}</a></td>'
            f'<td class="num">{v:.4f}</td><td class="num">{t:.4f}</td>'
            f'<td class="num delta {"drop" if v - t > 0 else ""}">{v - t:+.4f}</td>'
            f'<td class="gapcell"><span class="gaptrack">'
            f'<i class="seg" style="left:{min(x1, x2):.1f}%;'
            f'width:{abs(x2 - x1):.1f}%"></i>'
            f'<i class="pt test" style="left:{x2:.1f}%"></i>'
            f'<i class="pt val" style="left:{x1:.1f}%"></i>'
            f'</span></td></tr>')
    return f"""<h2>The val gain is capacity, not locality</h2>
<p class="note">Inner-val F1 (hollow) against held-out-region F1 (solid), for every
locality arm plus two references: the deployed baseline, and <code>ctrl_capacity</code>
— a plain MLP with no router at all, widened to the MoE's 78k parameters. Every
locality arm lifts val by 0.013 to 0.018 and moves the held-out region by nothing,
which looks like locality being learned and then failing to cross the fold boundary.
The control says otherwise: it reaches val 0.8121 on its own, and on fold 1 it beats
the MoE's val outright. Of the MoE's +0.0178 val gain, capacity owns +0.0147 and
routing +0.0031; on test the split is +0.0007 and +0.0016. Routing's share of both
sits inside the noise floor.</p>
<div class="scroll"><table>
<thead><tr><th>trial</th><th class="num">inner val</th><th class="num">held-out</th>
<th class="num">gap</th><th>val vs held-out</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def locality_panel():
    """What a local label is worth, as a budget curve.

    The MoE arms all have to synthesise locality from the training regions. This
    panel is the other half of the question: give the model real labels inside
    the held-out region and see what they buy. Bars are paired deltas against the
    same fold's global model, scored on evaluation cells no arm ever trains on.
    """
    p = ac.RESULTS_DIR / "diag_locality.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    arms = ["local_only", "finetune", "pooled"]
    label = {"local_only": "local model only",
             "finetune": "global, fine-tuned locally",
             "pooled": "retrained on both"}
    by = {(s["arm"], s["budget"]): s for s in d["summary"]}
    budgets = d["budgets"]
    vals = [by[(a, b)]["delta_vs_global"] for a in arms for b in budgets
            if (a, b) in by]
    if not vals:
        return ""
    lim = max(0.02, max(abs(v) for v in vals) * 1.15)
    groups = []
    for a in arms:
        bars = []
        for b in budgets:
            s = by.get((a, b))
            if not s:
                continue
            v = s["delta_vs_global"]
            h = min(abs(v) / lim, 1.0) * 46
            cls = "up" if v > 0 else "dn"
            tick = str(b)
            bars.append(
                f'<div class="lb"><div class="lbw">'
                f'<i class="{cls}" style="height:{h:.1f}px" '
                f'title="{v:+.4f} ± {s["sd"]:.4f}"></i></div>'
                f'<span class="lv">{v:+.3f}</span>'
                f'<span class="lx">{tick}</span></div>')
        groups.append(f'<div class="lgrp"><h4>{label[a]}</h4>'
                      f'<div class="lbars">{"".join(bars)}</div></div>')
    g = d.get("global_f1_per_fold", [])
    gtxt = (" · global model on the same rows: "
            + ", ".join(f"{v:.3f}" for v in g)) if g else ""
    rowsz = sorted({(s["budget"], s.get("n_local_mean", 0)) for s in d["summary"]})
    scale = " · ".join(f"{b} cell{'s' if b > 1 else ''} ≈ {n:,} rows"
                       for b, n in rowsz if n)
    return f"""<h2>What a local label is actually worth</h2>
<p class="note">Labels bought inside the held-out region, in whole cells because a
field crew visits places rather than pixels, and scored on evaluation cells that no
arm ever trains on. The x axis is cells surveyed ({esc(scale)}); bars are paired
against that fold's global model{esc(gtxt)}. Where the first panel crosses zero —
where a model trained on nothing but local labels overtakes one trained on 440k
labels from everywhere else — is the honest price of "local beats global".</p>
<div class="lpanel">{''.join(groups)}</div>"""


def floor_panel(recs):
    """The span the leaderboard lives inside: a linear probe, a tuned forest,
    the deployed net, the round's best arm, and the region oracle.

    Every delta on this page is measured against the deployed MLP, which can
    only ever answer "better than what we ship". It cannot say how much of the
    0.7414 the model is responsible for at all — and without that, three rounds
    of ties have no scale to be read on.
    """
    want = ["probe_linear", "rf_tuned", "baseline"]
    by = {r["name"]: r for r in recs}
    if not all(n in by for n in want):
        return ""
    scored = trial_recs(recs)
    best = max(scored, key=lambda r: r["delta_mean"]) if scored else None
    bound = by.get("bound_region_oracle")

    picks = [(by["probe_linear"], "linear probe on the same 67 features", "floor"),
             (by["rf_tuned"], "tuned random forest, same folds and weights", "floor"),
             (by["baseline"], "the deployed 5-seed MLP", "base")]
    if best:
        picks.append((best, f"best of {len(scored)} scored trials", "best"))
    if bound:
        picks.append((bound, "per-row oracle over regional models — not reachable",
                      "bound"))

    vals = [r["f1_mean"] for r, _, _ in picks]
    lo, hi = min(vals) - 0.004, max(vals) + 0.004
    span = hi - lo
    base_f1 = by["baseline"]["f1_mean"]
    rows = []
    for r, blurb, kind in picks:
        x = (r["f1_mean"] - lo) / span * 100
        # the PAIRED per-fold delta, not the difference of means — the same
        # number the leaderboard shows, and the only one this page trusts
        d = r.get("delta_mean", r["f1_mean"] - base_f1)
        rows.append(
            f'<tr class="{kind}">'
            f'<td class="nm"><a href="#{esc(r["name"])}">{esc(r["name"])}</a>'
            f'<span>{esc(blurb)}</span></td>'
            f'<td class="num">{r["f1_mean"]:.4f}</td>'
            f'<td class="num delta">{"—" if kind == "base" else f"{d:+.4f}"}</td>'
            f'<td class="axcell"><span class="axtrack">'
            f'<i class="tick" title="deployed baseline {base_f1:.4f}" '
            f'style="left:{(base_f1 - lo) / span * 100:.1f}%"></i>'
            f'<i class="dot {kind}" style="left:{x:.1f}%" '
            f'title="{esc(r["name"])}: {r["f1_mean"]:.4f} '
            f'(folds {", ".join(f"{v:.4f}" for v in r["f1_per_fold"])})"></i>'
            f'</span></td></tr>')

    lin = by["probe_linear"]["f1_mean"]
    rf = by["rf_tuned"]["f1_mean"]
    gain = -by["probe_linear"]["delta_mean"]        # paired, per fold
    contested = best["delta_mean"] if best else 0.0
    vgap_rf = by["rf_tuned"]["val_f1_mean"] - rf
    vgap_base = by["baseline"]["val_f1_mean"] - base_f1
    # the forest's in-distribution fit is enormous; where it lands on held-out
    # regions relative to a LINEAR model is the point, so state what happened
    # rather than what was expected
    lands = ("<em>below the linear probe</em>" if rf < lin else
             f"only {rf - lin:+.4f} past the linear probe")
    return f"""<h2>The floor this whole page sits on</h2>
<p class="note">Same folds, same cls12 relabel, same sqrt class weights, same
training-fold-only scaler — only the estimator changes. Gold tick marks the deployed
model; every dot is one model on one shared macro-F1 axis. A <b>linear probe</b> on
the 67 features reaches {lin:.4f} and a <b>tuned random forest</b> {rf:.4f}, so
everything the deployed network's two hidden layers buy over a linear read of
AlphaEarth is <b>{gain:+.4f}</b>. The best of {len(scored)} scored trials adds
<b>{contested:+.4f}</b> on top of that — {gain / max(contested, 1e-9):.0f}&times;
smaller than the nonlinearity itself, and inside the seed noise. That ratio is the
argument for stopping: the remaining headroom is not in the estimator.</p>
<p class="note">The forest is the more interesting of the two. It fits the inner
val to {by['rf_tuned']['val_f1_mean']:.4f} — far past the deployed net's
{by['baseline']['val_f1_mean']:.4f} — and then lands {lands}
on held-out regions, a val-to-test gap of {vgap_rf:.3f} against the net's
{vgap_base:.3f}. Axis-aligned splits memorise the training regions and do not
carry across the fold boundary, which is the same failure the capacity,
regularisation and locality studies each found from a different direction. It is
also the clearest statement of what the deployed MLP is for: not more fit, but a
decision surface smooth enough to survive being moved.</p>
<div class="scroll"><table class="floor">
<thead><tr><th>model</th><th class="num">macro-F1</th><th class="num">Δ deployed</th>
<th>{lo:.3f} &nbsp;&rarr;&nbsp; {hi:.3f}</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>"""


def cost_panel():
    """What each arm costs on the deployment path.

    The round ranked mechanisms on macro-F1 alone. `predict_raster.py` runs the
    chosen net over ~1.3 billion pixels, so the second column of that decision is
    wall time, and the number that decides it is not arm-vs-arm — it is arm
    against the raster I/O the pipeline is already bound by.
    """
    p = ac.RESULTS_DIR / "diag_inference_cost.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    arms = d["arms"]
    io_fast = d["io_px_per_s"]["local ssd, 6 readers"]
    io_min = d["aoi_px"] / io_fast / 60
    # Bars fill at most 96% of the track; the value is direct-labelled in its own
    # column instead of riding on the bar, so a long bar can never sit under its
    # own label.
    hi = max(max(a["aoi_minutes"] for a in arms), io_min) / 0.96
    io_x = io_min / hi * 100
    rows = []
    for a in arms:
        base = a["name"] == "baseline"
        w = a["aoi_minutes"] / hi * 100
        over = a["aoi_minutes"] > io_min
        rows.append(
            f'<tr class="{"base" if base else ""}">'
            f'<td class="nm"><a href="#{esc(a["name"])}">{esc(a["name"])}</a>'
            f'<span>{esc(a["label"])}</span></td>'
            f'<td class="num">{a["params"]:,}</td>'
            f'<td class="num">{a["macs_row_dense"]:,}</td>'
            f'<td class="num sub">{a["macs_row_routed_ideal"]:,}</td>'
            f'<td class="num">{a["px_per_s"] / 1e6:.2f}</td>'
            f'<td class="num">{a["aoi_minutes"]:.1f}<em>&thinsp;{a["rel_time"]:.1f}&times;</em></td>'
            f'<td class="costcell"><span class="costtrack" '
            f'title="{esc(a["name"])}: {a["aoi_minutes"]:.1f} min of GPU against a '
            f'{io_min:.0f} min read">'
            f'<i class="io" style="left:{io_x:.1f}%"></i>'
            f'<i class="cb {"base" if base else ""} {"over" if over else ""}" '
            f'style="width:{w:.1f}%"></i></span></td></tr>')
    base = next(a for a in arms if a["name"] == "baseline")
    moe8 = next((a for a in arms if "n_experts8" in a["name"]), None)
    moe16 = next((a for a in arms if "n_experts16" in a["name"]), None)

    # Where the same six arms land on the two other machines this actually runs
    # on. Both flip the conclusion, in opposite directions, which is why the
    # verdict is "free HERE" rather than "free".
    cifs_min = d["aoi_px"] / d["io_px_per_s"]["cifs P-drive"] / 60
    cpu = [a["cpu_px_per_s"] for a in arms if a.get("cpu_px_per_s")]
    elsewhere = (f"On the CIFS P-drive the read alone is {cifs_min:.0f} min and "
                 f"every arm here disappears behind it.")
    if cpu:
        elsewhere += (f" With no GPU at all the same six take "
                      f"{d['aoi_px'] / max(cpu) / 60:.0f}–"
                      f"{d['aoi_px'] / min(cpu) / 60:.0f} min of compute, and the "
                      f"choice stops being free anywhere.")
    return f"""<h2>What the winner would cost to run</h2>
<p class="note">Measured on the deployed path — <code>predict_classmap_gpu</code>,
5-seed ensemble, {d['chunk']:,}-row chunks on the {esc(d['device'])} — over the
wall-to-wall target of 1.3&nbsp;billion pixels. The dashed line is the pipeline's
read ceiling on this 8-core box ({io_fast / 1e6:.1f}&nbsp;M&nbsp;px/s with six
reader threads); <code>predict_raster.py</code> overlaps read with GPU, so the run
takes roughly the LONGER of the two and a bar left of the line costs nothing
end-to-end. The deployed MLP sits at {base['aoi_minutes']:.1f} min against a
{io_min:.0f}-min read — {io_min / base['aoi_minutes']:.1f}&times; of headroom, which
is why the README calls this pipeline I/O-bound.
<code>moe_shared@8&nbsp;experts</code>, the arm with the best mean delta, spends
{moe8['aoi_minutes']:.1f} min: still under the line, so on THIS box it is close to
free — and it has eaten {moe8['aoi_minutes'] / io_min * 100:.0f}&#37; of the
headroom, so the moment reads get faster (more cores, local NVMe, a cached AOI) the
model becomes the wall instead of the raster.
<code>@16&nbsp;experts</code> already crosses it at {moe16['aoi_minutes']:.1f} min.
The two MAC columns are the other half of the story: the second is what an ideal
sparse kernel would execute, the first is what the code runs, and the difference is
work the dense expert loop never skips. Wall time is worse than even that —
{moe8['rel_time']:.1f}&times; the baseline for {moe8['rel_macs']:.1f}&times; the
MACs — because the experts are narrow and each re-reads the whole input block, so
they run far below the arithmetic intensity of the trunk. Top-k here buys capacity,
not FLOPs.</p>
<div class="scroll"><table class="cost">
<thead><tr><th>arm</th><th class="num">params</th><th class="num">MAC/row</th>
<th class="num sub">ideal sparse</th><th class="num">M px/s</th>
<th class="num">min / 1.3B px</th><th>vs the read ceiling</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<p class="clegend"><i class="sw bar"></i>GPU time, 5-seed ensemble
<i class="sw io"></i>read ceiling ({io_min:.0f} min at {io_fast / 1e6:.1f} M px/s)
<i class="sw over"></i>model is now the bottleneck
<span>{esc(elsewhere)}</span></p>"""


def build():
    recs = load()
    base = next((r for r in recs if r["name"] == "baseline"), None)
    scored = trial_recs(recs)
    wins = [r for r in scored if r.get("verdict") == "WIN"]
    clean = [r for r in wins if r.get("no_tradeoff")]
    best = max(scored, key=lambda r: r["delta_mean"]) if scored else None
    oracle = next((r for r in recs if r["name"] == "dr_oracle_offsets"), None)

    # The pre-registered queues are the PLAN; the sweeps, combos and replications
    # each round generated from its own results are not in them, so the plan size
    # stopped being a denominator several rounds ago (it read "69/44"). Report
    # what was actually scored.
    state = ac.RESULTS_DIR / "_loop_state.json"
    running = state.exists() and (time.time() - state.stat().st_mtime) < 1800

    tiles = [
        ("deployed baseline", f"{base['f1_mean']:.4f}" if base else "—",
         "macro-F1, 3-fold spatial CV" if base else "not yet run"),
        ("trials scored", f"{len(scored)}",
         "one mechanism each, plus the sweeps and combos they generated"),
        ("best delta so far",
         f"{best['delta_mean']:+.4f}" if best else "—",
         esc(best["name"]) if best else "—"),
        ("wins", f"{len(wins)}",
         "all folds positive AND Δ&gt;0.003" +
         (" — and the best result halved on a fresh seed" if not wins else "")),
    ]
    if oracle:
        tiles.append(("decision-rule headroom", f"{oracle['delta_mean']:+.4f}",
                      "upper bound if the rule were fitted on the test fold"))
    # The scale everything else is read on: a linear model on the same features.
    probe = next((r for r in recs if r["name"] == "probe_linear"), None)
    if probe:
        tiles.append(("linear-probe floor", f"{probe['f1_mean']:.4f}",
                      f"the deep net is worth {-probe['delta_mean']:+.4f} over it"))

    tile_html = "".join(
        f'<div class="tile"><span class="lab">{lab}</span>'
        f'<span class="val">{val}</span><span class="sub">{sub}</span></div>'
        for lab, val, sub in tiles)

    prior_html = "".join(
        f"<li><b>{k}</b><span>{v}</span></li>" for k, v in PRIOR)

    page = TEMPLATE.format(
        updated=time.strftime("%Y-%m-%d %H:%M"),
        status=("running" if running else "idle"),
        tiles=tile_html,
        leaderboard=leaderboard(recs),
        floor=floor_panel(recs),
        cost=cost_panel(),
        routing=routing_panel(recs),
        gap=gap_panel(recs),
        locality=locality_panel(),
        matrix=class_panel(recs),
        log=log_cards(recs),
        prior=prior_html,
        findings=findings_html(),
        wandb=WANDB_URL,
        n_done=len(scored),
    )
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(recs)} records)")


TEMPLATE = """<title>Weak-class autoresearch — nyvest land-cover DNN</title>
<style>
:root {{
  --paper:#e9ebe6; --card:#f3f4f0; --ink:#161c1a; --slate:#5c6a66;
  --rule:#ccd1cb; --gold:#8a5f10; --gold-soft:#c9a227;
  --moss:#3f6b46; --rust:#9c4020; --shadow:rgba(22,28,26,.07);
  --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
  --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root {{
    --paper:#121614; --card:#1a201e; --ink:#e4e8e3; --slate:#94a29d;
    --rule:#2c3532; --gold:#d3a53c; --gold-soft:#8a6a1c;
    --moss:#7bb083; --rust:#d6795a; --shadow:rgba(0,0,0,.4);
  }}
}}
:root[data-theme="dark"] {{
  --paper:#121614; --card:#1a201e; --ink:#e4e8e3; --slate:#94a29d;
  --rule:#2c3532; --gold:#d3a53c; --gold-soft:#8a6a1c;
  --moss:#7bb083; --rust:#d6795a; --shadow:rgba(0,0,0,.4);
}}
:root[data-theme="light"] {{
  --paper:#e9ebe6; --card:#f3f4f0; --ink:#161c1a; --slate:#5c6a66;
  --rule:#ccd1cb; --gold:#8a5f10; --gold-soft:#c9a227;
  --moss:#3f6b46; --rust:#9c4020; --shadow:rgba(22,28,26,.07);
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--serif); font-size:16px; line-height:1.6;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:clamp(1.5rem,4vw,3.5rem) clamp(1rem,4vw,2.5rem) 5rem; }}
a {{ color:inherit; }}
a:focus-visible, summary:focus-visible {{ outline:2px solid var(--gold); outline-offset:3px; }}

header.masthead {{ border-bottom:2px solid var(--ink); padding-bottom:1.4rem; }}
.eyebrow {{
  font-family:var(--mono); font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--slate); margin:0 0 .9rem;
  display:flex; gap:1rem; flex-wrap:wrap; align-items:center;
}}
.dot {{ width:.5rem; height:.5rem; border-radius:50%; background:var(--gold); display:inline-block; }}
.dot.idle {{ background:var(--slate); }}
h1 {{
  font-size:clamp(1.9rem,4.5vw,3.1rem); line-height:1.08; margin:0 0 .6rem;
  font-weight:600; letter-spacing:-.02em; text-wrap:balance; max-width:20ch;
}}
.dek {{ margin:0; max-width:64ch; color:var(--slate); font-size:1.05rem; }}

.tiles {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); margin:2.2rem 0 3rem; }}
.tile {{ background:var(--card); padding:1rem 1.1rem; display:flex; flex-direction:column; gap:.15rem; }}
.tile .lab {{ font-family:var(--mono); font-size:.66rem; letter-spacing:.13em;
  text-transform:uppercase; color:var(--slate); }}
.tile .val {{ font-family:var(--mono); font-size:1.75rem; font-variant-numeric:tabular-nums;
  letter-spacing:-.02em; }}
.tile .val em {{ font-size:1rem; font-style:normal; color:var(--slate); }}
.tile .sub {{ font-size:.8rem; color:var(--slate); line-height:1.35; }}

h2 {{ font-size:1.35rem; font-weight:600; letter-spacing:-.01em; margin:3rem 0 .35rem;
  padding-top:1.6rem; border-top:1px solid var(--rule); }}
h2:first-of-type {{ border-top:0; padding-top:0; }}
.note {{ margin:.2rem 0 1.4rem; color:var(--slate); font-size:.93rem; max-width:70ch; }}
.note code {{ font-family:var(--mono); font-size:.85em; background:var(--card);
  padding:.05em .35em; border:1px solid var(--rule); }}

.scroll {{ overflow-x:auto; border:1px solid var(--rule); background:var(--card); }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
thead th {{ font-family:var(--mono); font-size:.65rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--slate); font-weight:400; text-align:left;
  padding:.7rem .8rem; border-bottom:1px solid var(--rule); white-space:nowrap; }}
tbody td {{ padding:.55rem .8rem; border-bottom:1px solid var(--rule); vertical-align:middle; }}
tbody tr:last-child td {{ border-bottom:0; }}
tbody tr:hover {{ background:color-mix(in srgb, var(--gold) 7%, transparent); }}
tr.base {{ background:color-mix(in srgb, var(--gold) 11%, transparent); }}
tr.base .nm a {{ font-weight:600; }}
.nm {{ font-family:var(--mono); font-size:.82rem; }}
.nm a {{ text-decoration:none; border-bottom:1px solid var(--rule); }}
.nm a:hover {{ border-color:var(--gold); }}
td.tier {{ font-size:.78rem; color:var(--slate); }}
.num {{ font-family:var(--mono); font-variant-numeric:tabular-nums; text-align:right;
  white-space:nowrap; }}
.num em {{ font-style:normal; color:var(--slate); font-size:.78em; }}
.delta.gain {{ color:var(--moss); font-weight:600; }}
.delta.drop {{ color:var(--rust); }}
.delta.flat {{ color:var(--slate); }}
.ctrl {{ color:var(--slate); font-size:.82rem; }}

.folds {{ display:inline-flex; gap:3px; }}
.sq {{ width:12px; height:12px; display:block; border:1px solid var(--rule); }}
.sq.pos {{ background:var(--moss); border-color:var(--moss); }}
.sq.neg {{ background:var(--rust); border-color:var(--rust); }}
.sq.nil {{ background:var(--rule); }}

.tradeoff {{ position:relative; display:grid; width:40px; height:26px;
  grid-template-columns:1fr 1fr; grid-template-rows:13px 13px; column-gap:4px; }}
.tradeoff .axis {{ position:absolute; left:0; right:0; top:13px; height:1px;
  background:var(--rule); }}
.tradeoff .bar {{ width:100%; height:var(--h); min-height:1px; }}
.tradeoff .weak {{ grid-column:1; }}
.tradeoff .strong {{ grid-column:2; }}
.tradeoff .bar.up {{ grid-row:1; align-self:end; }}
.tradeoff .bar.down {{ grid-row:2; align-self:start; }}
.tradeoff .weak {{ background:var(--moss); }}
.tradeoff .strong {{ background:var(--gold-soft); }}
.tradeoff .bar.down.weak {{ background:var(--rust); }}
.tradeoff .bar.down.strong {{ background:var(--rust); }}
.tradeoff.empty {{ opacity:.25; }}

/* val-vs-test dumbbell */
td.gapcell {{ width:32%; min-width:150px; }}
.gaptrack {{ position:relative; display:block; height:12px; }}
.gaptrack::before {{ content:""; position:absolute; left:0; right:0; top:5.5px;
  height:1px; background:var(--rule); }}
.gaptrack .seg {{ position:absolute; top:5px; height:2px; background:var(--gold-soft); }}
.gaptrack .pt {{ position:absolute; top:2px; width:8px; height:8px; margin-left:-4px;
  border-radius:50%; }}
.gaptrack .pt.val {{ background:var(--paper); border:1.5px solid var(--slate); }}
.gaptrack .pt.test {{ background:var(--moss); }}

/* floor panel: every model on one shared macro-F1 axis */
table.floor td.nm span, table.cost td.nm span {{ display:block; font-family:var(--serif);
  font-style:italic; font-size:.78rem; color:var(--slate); margin-top:.15rem; }}
table.floor tr.base td {{ background:color-mix(in srgb, var(--gold-soft) 9%, transparent); }}
td.axcell {{ width:40%; min-width:170px; }}
.axtrack {{ position:relative; display:block; height:14px; }}
.axtrack::before {{ content:""; position:absolute; left:0; right:0; top:6.5px;
  height:1px; background:var(--rule); }}
.axtrack .tick {{ position:absolute; top:0; width:1px; height:14px;
  background:var(--gold); opacity:.55; }}
.axtrack .dot {{ position:absolute; top:2.5px; width:9px; height:9px; margin-left:-4.5px;
  border-radius:50%; background:var(--slate);
  box-shadow:0 0 0 2px var(--card); }}
.axtrack .dot.base {{ background:var(--gold); }}
.axtrack .dot.best {{ background:var(--moss); }}
/* the oracle is not achievable — hollow, so it never reads as a result */
.axtrack .dot.bound {{ background:var(--card); border:1.5px dashed var(--slate); }}

/* inference cost: minutes per 1.3B px against the pipeline's read ceiling */
td.costcell {{ width:34%; min-width:190px; }}
.costtrack {{ position:relative; display:block; height:18px; }}
.costtrack .cb {{ position:absolute; left:0; top:5px; height:8px; border-radius:0 4px 4px 0;
  background:var(--slate); opacity:.55; }}
.costtrack .cb.base {{ background:var(--gold); opacity:.75; }}
.costtrack .cb.over {{ background:var(--rust); opacity:.7; }}
.costtrack .io {{ position:absolute; top:0; width:0; height:18px;
  border-left:1.5px dashed var(--moss); }}
table.cost td.num em {{ display:block; font-family:var(--mono); font-size:.6rem;
  font-style:normal; color:var(--slate); }}
table.cost th.sub, table.cost td.sub {{ color:var(--slate); }}
.clegend {{ font-family:var(--mono); font-size:.66rem; color:var(--slate);
  margin:.5rem 0 0; display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; }}
.clegend i.sw {{ display:inline-block; width:14px; height:8px; border-radius:0 3px 3px 0;
  margin-left:.9rem; }}
.clegend i.sw:first-child {{ margin-left:0; }}
.clegend i.sw.bar {{ background:var(--slate); opacity:.55; }}
.clegend i.sw.over {{ background:var(--rust); opacity:.7; }}
.clegend i.sw.io {{ width:0; height:13px; border-radius:0;
  border-left:1.5px dashed var(--moss); }}
.clegend span {{ flex-basis:100%; font-family:var(--serif); font-style:italic;
  font-size:.8rem; margin-top:.4rem; }}
table.cost tr.base td {{ background:color-mix(in srgb, var(--gold-soft) 9%, transparent); }}

/* label-budget panel: one small multiple per adaptation strategy, shared scale */
.lpanel {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); margin:1.2rem 0; }}
.lgrp {{ background:var(--card); padding:1rem 1.1rem 1.2rem; }}
.lgrp h4 {{ font-family:var(--mono); font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--gold); margin:0 0 1rem; font-weight:600; }}
.lbars {{ display:flex; align-items:center; gap:.55rem; }}
.lb {{ flex:1; display:flex; flex-direction:column; align-items:center; }}
.lbw {{ height:96px; width:100%; display:grid; grid-template-rows:1fr 1fr;
  position:relative; }}
.lbw::after {{ content:""; position:absolute; left:0; right:0; top:48px; height:1px;
  background:var(--rule); }}
.lbw i {{ width:100%; min-height:1px; }}
.lbw i.up {{ grid-row:1; align-self:end; background:var(--moss); }}
.lbw i.dn {{ grid-row:2; align-self:start; background:var(--rust); }}
.lv {{ font-family:var(--mono); font-size:.6rem; color:var(--slate); margin-top:.4rem; }}
.lx {{ font-family:var(--mono); font-size:.6rem; color:var(--slate); opacity:.7; }}

.chip {{ font-family:var(--mono); font-size:.62rem; letter-spacing:.1em;
  text-transform:uppercase; padding:.22em .5em; border:1px solid currentColor;
  white-space:nowrap; }}
.chip.win {{ color:var(--moss); }}
.chip.loss {{ color:var(--rust); }}
.chip.tie {{ color:var(--slate); }}
.chip.ref {{ color:var(--gold); }}
.chip.bound {{ color:var(--slate); border-style:dashed; }}
.chip.split {{ color:var(--gold); border-style:dashed; }}
.nt {{ font-family:var(--mono); font-size:.6rem; letter-spacing:.09em;
  color:var(--gold); text-transform:uppercase; }}
.flag {{ font-family:var(--mono); font-size:.58rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--slate); border:1px dotted var(--rule);
  padding:.1em .35em; margin-left:.4rem; white-space:nowrap; }}

.findings {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); }}
.finding {{ background:var(--card); padding:1.1rem 1.2rem 1.2rem;
  border-top:3px solid var(--gold); }}
.finding.method {{ border-top-color:var(--rust); }}
.finding h4 {{ margin:0 0 .5rem; font-size:1.02rem; font-weight:600;
  letter-spacing:-.01em; text-wrap:balance; }}
.finding p {{ margin:0; font-size:.9rem; color:var(--slate); }}

.matrix th.cls {{ text-align:left; font-family:var(--serif); font-weight:400;
  white-space:nowrap; padding:.5rem .8rem; border-bottom:1px solid var(--rule); }}
.matrix th.cls b {{ font-weight:600; }}
.matrix th.cls span {{ font-family:var(--mono); font-size:.7rem; color:var(--slate);
  margin-left:.45rem; }}
.matrix th.cls.weak {{ border-left:3px solid var(--rust); }}
.matrix th.cls.strong {{ border-left:3px solid var(--rule); }}
.matrix td.base {{ font-weight:600; }}
.matrix th.colhead {{ font-family:var(--mono); font-size:.62rem; text-align:right;
  max-width:78px; white-space:normal; word-break:break-word; }}
.cell {{ font-family:var(--mono); font-size:.74rem; text-align:right;
  font-variant-numeric:tabular-nums; }}
.cell.p {{ background:color-mix(in srgb, var(--moss) calc(var(--a)*55%), transparent); }}
.cell.n {{ background:color-mix(in srgb, var(--rust) calc(var(--a)*55%), transparent); }}
.cell.z {{ color:var(--slate); }}
/* the classical-floor columns: same measure, same scale, own group */
.matrix .edge {{ border-left:2px solid var(--ink); }}
.matrix th.colhead.flr span {{ color:var(--gold); }}
.matrix tr.grp td {{ border-bottom:0; padding:.45rem .8rem .1rem; }}
.matrix td.gl {{ font-family:var(--mono); font-size:.6rem; letter-spacing:.11em;
  text-transform:uppercase; color:var(--slate); text-align:left; white-space:nowrap; }}

section.tier-group h3 {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--gold); margin:2.2rem 0 .3rem; }}
.blurb {{ margin:0 0 1rem; color:var(--slate); font-size:.9rem; max-width:72ch; }}
.cards {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); }}
.card {{ background:var(--card); padding:1rem 1.1rem 1.1rem; }}
.card header {{ display:flex; align-items:center; justify-content:space-between;
  gap:.6rem; margin-bottom:.5rem; }}
.card h4 {{ font-family:var(--mono); font-size:.86rem; margin:0; font-weight:600; }}
.card .idea {{ margin:0 0 .6rem; font-size:.93rem; }}
.card .hyp {{ margin:0 0 .55rem; font-size:.88rem; color:var(--slate); }}
.card .hyp span {{ display:block; font-family:var(--mono); font-size:.6rem;
  letter-spacing:.12em; text-transform:uppercase; color:var(--gold);
  margin-bottom:.15rem; }}
.card .prov {{ margin:0 0 .55rem; font-size:.8rem; color:var(--slate); font-style:italic; }}
.card .res {{ margin:0; font-family:var(--mono); font-size:.74rem;
  font-variant-numeric:tabular-nums; color:var(--slate);
  border-top:1px solid var(--rule); padding-top:.5rem; }}

ul.prior {{ list-style:none; margin:0; padding:0; display:grid; gap:1px;
  background:var(--rule); border:1px solid var(--rule); }}
ul.prior li {{ background:var(--card); padding:.75rem 1rem; display:grid;
  grid-template-columns:minmax(140px,190px) 1fr; gap:1rem; font-size:.88rem; }}
ul.prior b {{ font-family:var(--mono); font-size:.72rem; letter-spacing:.06em;
  text-transform:uppercase; color:var(--gold); font-weight:400; }}
ul.prior span {{ color:var(--slate); }}
@media (max-width:600px) {{ ul.prior li {{ grid-template-columns:1fr; gap:.2rem; }} }}

footer {{ margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--rule);
  font-family:var(--mono); font-size:.72rem; color:var(--slate);
  display:flex; gap:1.2rem; flex-wrap:wrap; }}
footer a {{ color:var(--gold); }}
</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow"><span><span class="dot {status}"></span> loop {status}</span>
     <span>updated {updated}</span>
     <span>{n_done} trials scored</span></p>
  <h1>Lifting the weak classes without paying for it</h1>
  <p class="dek">An autonomous search for macro-F1 above the deployed land-cover
  DNN, judged on a stricter bar than usual: a mechanism only counts if it gains on
  every spatial fold <em>and</em> does not buy weak-class accuracy out of the
  strong classes. {n_done} mechanisms have now been scored against it — margins,
  priors, hierarchy, oversampling, distillation, modern optimizers and
  activations, new architectures, and transductive adaptation to the target
  region. None wins, and the closest thing to one halved when re-run on a fresh
  seed.</p>
</header>

<div class="tiles">{tiles}</div>

<h2>Leaderboard</h2>
<p class="note">Sorted by paired delta against the baseline's own per-fold scores.
Fold spread (~0.02) dwarfs the effects being chased (~0.003), so run means are
meaningless — the three squares are the per-fold signs, and a
<code>WIN</code> needs all three green plus Δ&gt;0.003. The bar to the right is the
tradeoff: weak-class mean gain above the axis, worst strong-class change below.
Gold below the axis is a class being paid out of. Trials that hold their inner val
out by whole cells pay a 0.0042 tax for that alone, so they carry a second delta
against the matched <code>baseline_gval</code> control — that column is the honest
one for them.</p>
<div class="scroll"><table>
<thead><tr><th>trial</th><th>tier</th><th class="num">macro-F1</th>
<th class="num">Δ paired</th><th class="num">Δ vs control</th><th>folds</th><th>weak / strong</th><th>verdict</th></tr></thead>
<tbody>
{leaderboard}
</tbody></table></div>

{floor}

{findings}

{cost}

{gap}

{routing}

{locality}

<h2>Where each class moved</h2>
<p class="note">Baseline per-class F1, weakest first. Red left edge marks the four
classes this search is for. A column that is green at the top and red at the bottom
is a trade, not a win. Every cell is the paired Δ against the baseline's own
per-class score, all on one colour scale — which is why the two floor columns are
solid and the six trial columns are nearly blank. The deep net's margin over a
linear probe is largest on exactly the classes the search is for: rock/sand
&minus;0.087, grassland &minus;0.054, wetland &minus;0.047. The best weak-class
gain anywhere in the right-hand group is +0.005 on scrub, against a
&minus;0.032 probe deficit on that same class. The forest is not a uniformly worse
probe — it is <em>better</em> on rock/sand (&minus;0.045) and scrub
(&minus;0.013) and then collapses on grassland (&minus;0.103) and snow/ice
(&minus;0.131), the two classes whose boundaries are least axis-aligned. Snow/ice
is also the one class the search does move (+0.02 to +0.05 across the group), so
the single thing eleven locality mechanisms found is the single thing a forest gets
most wrong.</p>
{matrix}

<h2>The log</h2>
<p class="note">Every mechanism, what it changes, and why it might beat a model
whose ceiling has already survived capacity, data, regularisation, routing and
ensembling. Hypotheses are recorded before the run, so a negative result still
buys information.</p>
{log}

<h2>Already ruled out</h2>
<p class="note">Prior work on this model. The search space below is what remains
after these, which is why nothing here is another capacity or regularisation
sweep.</p>
<ul class="prior">{prior}</ul>

<footer>
  <span>DNN/autoresearch</span>
  <span>3-fold GroupKFold on cell_id · 664k rows · 64 AlphaEarth + 3 lidar</span>
  <a href="{wandb}">every run logged to W&amp;B</a>
</footer>
</div>
"""

if __name__ == "__main__":
    build()
