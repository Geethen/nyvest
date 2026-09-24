"""ar3 trial registry. Every arm inherits the deployed moe8 recipe (ar3_common)
and varies ONE thing. Pre-registered queue in PLAN.md."""

from __future__ import annotations

import dataclasses

import ar3_common as A

Trial3 = A.Trial3
TRIALS: dict[str, Trial3] = {}


def _add(t):
    TRIALS[t.name] = t


_add(Trial3(name="baseline", tier="baseline",
            idea="the deployed moe8 recipe, unchanged",
            hypothesis="reference for every paired delta"))
_add(Trial3(name="baseline_s1", tier="baseline",
            idea="the deployed moe8 recipe on an INDEPENDENT seed set (run with AR_SEED=1)",
            hypothesis="its paired delta against `baseline` IS the noise floor of every metric"))

# ---- consensus labels in error-prone areas (W1 data, data/consensus/) ----
HARD = ["flip", "uncertain"]
_add(Trial3(name="cons_add", tier="consensus",
            idea="+ hard-area rows (flip/uncertain strata, TRAIN cells) whose grunnkart "
                 "label is backed by >=2 external products (gk_plus2)",
            hypothesis="the FSCS frame never samples where the maps flip; labels there "
                       "should fix exactly the pixels that produce spurious change",
            extra={"rule": "gk_plus2", "strata": HARD, "weight": 1.0}))
_add(Trial3(name="cons_add_w05", tier="consensus", idea="cons_add at row weight 0.5",
            hypothesis="consensus labels are noisier than FSCS; down-weighting keeps "
                       "their coverage without letting them dominate",
            extra={"rule": "gk_plus2", "strata": HARD, "weight": 0.5}))
_add(Trial3(name="ctrl_hard_gk", tier="consensus",
            idea="control: the SAME strata, raw grunnkart labels, same row count as cons_add",
            hypothesis="separates 'labels in hard areas' from 'the consensus filter'",
            extra={"rule": "gk_raw", "strata": HARD, "match_n_of": "cons_add"}))
_add(Trial3(name="ctrl_rand_cons", tier="consensus",
            idea="control: consensus rows from the RANDOM stratum, same row count as cons_add",
            hypothesis="separates 'hard areas' from 'more consensus data'",
            extra={"rule": "gk_plus2", "strata": ["random"], "match_n_of": "cons_add"}))
_add(Trial3(name="cons_ext_majority", tier="consensus",
            idea="+ hard-area rows labelled by the external-product majority (grunnkart ignored)",
            hypothesis="if grunnkart is what is wrong in hard areas, the products' own "
                       "majority is the better teacher (definition mismatch is the risk)",
            extra={"rule": "ext_majority", "strata": HARD, "weight": 1.0}))


def resolve(spec: str) -> Trial3:
    """`name` or `name@k=v,...` (config overrides; `n_ensemble`/`label_smooth`
    are Trial fields and are set directly)."""
    if spec in TRIALS:
        return TRIALS[spec]
    if spec.startswith("combo:"):
        # hooks must not collide, else the combo is a third, unnamed mechanism
        parts = [resolve(x) for x in spec[len("combo:"):].split("+")]
        t = dataclasses.replace(parts[0])
        cfg = dict(t.config)
        for o in parts[1:]:
            for h in ("build_fn", "loss_fn", "opt_fn", "after_fit", "fold_fn",
                      "feat_fn", "prep_fn", "post3_fn", "extra"):
                theirs = getattr(o, h)
                if theirs is None:
                    continue
                if getattr(t, h) is not None:
                    raise SystemExit(f"cannot combine {spec}: both set {h}")
                setattr(t, h, theirs)
            if o.label_smooth != A.ac.LABEL_SMOOTH:
                t.label_smooth = o.label_smooth
            cfg.update(o.config)
        t.config, t.tier = cfg, "combo"
        t.name = "combo_" + "_".join(p.name for p in parts)
        t.idea = " + ".join(p.idea for p in parts)
        t.hypothesis = "additive or redundant?"
        return t
    name, _, kv = spec.partition("@")
    t = dataclasses.replace(TRIALS[name])
    cfg = dict(t.config)
    for pair in kv.split(","):
        k, v = pair.split("=")
        v = float(v) if any(ch in v for ch in ".e") or v.lstrip("-").isdigit() else v
        if k in ("n_ensemble",):
            setattr(t, k, int(v))
        elif k in ("label_smooth",):
            setattr(t, k, float(v))
        else:
            cfg[k] = v
    t.config = cfg
    t.name = spec.replace("@", "__").replace("=", "").replace(",", "_").replace(".", "p")
    return t


try:  # temporal / calibration / consensus arms live in their own module
    import trials3_arms  # noqa: F401,E402
    trials3_arms.register(_add, Trial3)
except ImportError as e:
    if "trials3_arms" not in str(e):
        raise
