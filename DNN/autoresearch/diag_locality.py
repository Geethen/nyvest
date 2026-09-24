"""How much is a LOCAL label worth? — the diagnostic behind the locality round.

Every MoE arm in this round has to build locality out of the training regions,
because under GroupKFold a test fold has no labels at all. That is a real
constraint of the deployment, but it is not the whole of the claim "local models
are better than global models for local conditions". The rest of the claim is
about what happens when you DO have local labels, and that is a question about
survey budget rather than architecture.

So this measures it directly. Inside each held-out region:

  * `eval` cells are fixed up front and never trained on by anything, so all
    arms and all budgets are scored on identical rows.
  * `pool` cells supply the label budget — whole cells, because a field campaign
    visits places, not random pixels, and sampling rows would leak a cell's
    neighbours into training.

Four arms at each budget n:

  global      the deployed model. n = 0 local labels, 440k distant ones.
  local_only  an identically-shaped MLP trained on the n local labels ALONE.
              Where this crosses `global` is the honest price of the claim: it
              is the number of local labels worth more than the entire rest of
              Norway.
  finetune    the global model, fine-tuned on the n local labels.
  pooled      retrained from scratch on train fold + n local labels — what you
              would actually deploy, and the arm that says whether local labels
              are best spent adapting or just added to the pile.

This is NOT a trial: there is no delta against the baseline to report, because
every arm here sees labels the baseline is not allowed. It is the study that
tells the Vestland/Møre rollout what a field campaign buys.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import ar_common as ac
import data_utils as du
from ar_common import Trial

OUT = ac.RESULTS_DIR / "diag_locality.json"
# Budget is counted in CELLS, not rows. A cell is the unit a survey actually
# buys — a crew visits places — and it is also the unit that keeps this
# leak-free, since sampling rows would put a cell's own neighbours on both
# sides of the split. Each cell here is worth roughly 1.5-5k labelled rows.
BUDGETS = [int(v) for v in os.environ.get("DIAG_BUDGETS", "1,2,4,8,16,32").split(",")]
N_REPS = int(os.environ.get("DIAG_REPS", "2"))    # draws of which cells the crew visited
N_SEEDS = int(os.environ.get("DIAG_SEEDS", "3"))  # ensemble members per arm
FT_LR = 3e-4
FT_EPOCHS = 60
FT_PATIENCE = 10


def _macro(y_true, y_pred, keep):
    """Macro-F1 over the classes PRESENT in the evaluation rows.

    The global metric averages over all ten classes and scores an absent class
    zero. Inside one region several classes genuinely do not occur, and which
    ones differ by fold, so averaging over all ten would mix "this arm is worse"
    with "this region has no snow". The class set is computed once per fold from
    the eval rows and held fixed across every arm and budget."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=keep, average="macro",
                          zero_division=0))


def _bare(n_ens=N_SEEDS):
    return Trial(name="_arm", tier="", idea="", hypothesis="", n_ensemble=n_ens)


def _ctx(Xtr, ytr, Xval, yval, n_classes, in_dim):
    return {"in_dim": in_dim, "n_classes": n_classes,
            "Xtr_t": torch.tensor(Xtr, device=ac.DEVICE),
            "ytr_t": torch.tensor(ytr, device=ac.DEVICE),
            "Xval_t": torch.tensor(Xval, device=ac.DEVICE),
            "yval_np": yval,
            "w": ac.class_weights(ytr, n_classes, "sqrt"),
            "trial": _bare()}


def _split(n, rng, frac=0.1, floor=32):
    n_val = max(floor, int(n * frac))
    perm = rng.permutation(n)
    return perm[:n_val], perm[n_val:]


def _predict(models, X, n_classes):
    Xt = torch.tensor(X, device=ac.DEVICE)
    P = np.zeros((X.shape[0], n_classes))
    for m in models:
        P += ac.predict_probs(m, Xt, n_classes)
    del Xt
    torch.cuda.empty_cache()
    return P / len(models)


def _train(ctx, n_ens=N_SEEDS):
    models = []
    for e in range(n_ens):
        m, _ = ac.train_member(ctx["trial"], ctx, ac.SEED + 100 * e)
        models.append(m)
    return models


def _finetune(models, ctx):
    """Adapt each global member to the local labels, early-stopping on a local
    holdout — the practitioner's move, and the one that risks catastrophic
    forgetting if the budget is tiny."""
    out = []
    crit = nn.CrossEntropyLoss(weight=ctx["w"], label_smoothing=ac.LABEL_SMOOTH)
    for i, src in enumerate(models):
        # fine-tune a COPY: the global members are reused at every budget, so
        # adapting them in place would make each budget start from the last one
        m = ac.default_mlp(ctx["in_dim"], ctx["n_classes"], ctx).to(ac.DEVICE)
        m.load_state_dict(src.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=FT_LR, weight_decay=ac.WEIGHT_DECAY)
        g = torch.Generator(device=ac.DEVICE).manual_seed(ac.SEED + 100 * i)
        n_tr = ctx["Xtr_t"].shape[0]
        best, best_state, bad = -1.0, None, 0
        for _ in range(FT_EPOCHS):
            m.train()
            order = torch.randperm(n_tr, device=ac.DEVICE, generator=g)
            for j in range(0, n_tr, ac.BATCH):
                b = order[j:j + ac.BATCH]
                opt.zero_grad()
                crit(m(ctx["Xtr_t"][b]), ctx["ytr_t"][b]).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                vp = m(ctx["Xval_t"]).argmax(1).cpu().numpy()
            f1 = du.macro_f1(ctx["yval_np"], vp, ctx["n_classes"])
            if f1 > best + 1e-4:
                best, bad = f1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= FT_PATIENCE:
                    break
        m.load_state_dict(best_state)
        out.append(m)
    return out


def main():
    t0 = time.perf_counter()
    data = du.load_data(extra_features="lidar")
    X, y_enc, groups, df = data["X"], data["y_enc"], data["groups"], data["df"]
    classes = data["classes"]
    n_classes = len(classes)
    cls12 = classes.index(12) if 12 in classes else -1
    lon, lat = df["lon"].values, df["lat"].values
    if ac.RELABEL != "none":
        y_enc, _ = du.apply_cls12_relabel(y_enc, classes, ac.RELABEL)
    print(f"loaded {X.shape[0]:,} rows / {X.shape[1]} feats")

    rows = []
    for k, tr, te in du.fold_indices(y_enc, groups):
        rng = np.random.default_rng(1000 + k)
        keep = du.clean_stale_class_mask(X[tr], y_enc[tr], df.iloc[tr], cls12,
                                         lon[tr], lat[tr])
        tr_use = tr[keep]
        scaler = StandardScaler().fit(X[tr_use])
        Xtr = scaler.transform(X[tr_use]).astype(np.float32)
        ytr = y_enc[tr_use]

        # fixed evaluation half of the held-out region
        cells = np.unique(groups[te])
        perm = rng.permutation(len(cells))
        ev_cells = set(cells[perm[:len(cells) // 2]].tolist())
        is_ev = np.array([g in ev_cells for g in groups[te]])
        ev, pool = te[is_ev], te[~is_ev]
        Xev = scaler.transform(X[ev]).astype(np.float32)
        yev = y_enc[ev]
        present = sorted(set(yev.tolist()))
        print(f"\n=== fold {k}: {len(cells)} cells -> eval {len(ev):,} rows "
              f"({len(present)} classes present), pool {len(pool):,} rows ===")

        v_idx, t_idx = _split(len(Xtr), rng)
        gctx = _ctx(Xtr[t_idx], ytr[t_idx], Xtr[v_idx], ytr[v_idx], n_classes,
                    Xtr.shape[1])
        gmodels = _train(gctx)
        f1_global = _macro(yev, _predict(gmodels, Xev, n_classes).argmax(1), present)
        print(f"  global (0 local labels): {f1_global:.4f}")
        rows.append({"fold": k, "budget": 0, "rep": 0, "arm": "global",
                     "f1": round(f1_global, 4), "n_local": 0,
                     "n_eval": int(len(ev))})
        for kk in ("Xtr_t", "ytr_t", "Xval_t"):
            gctx.pop(kk, None)
        torch.cuda.empty_cache()

        pool_cells = np.unique(groups[pool])
        for budget in BUDGETS:
            for rep in range(N_REPS):
                r2 = np.random.default_rng(10_000 * k + 100 * rep + budget)
                if budget > len(pool_cells):
                    continue
                take = pool_cells[r2.permutation(len(pool_cells))[:budget]]
                sel = pool[np.isin(groups[pool], take)]
                if len(sel) < 64:
                    continue
                n_cls_local = len(set(y_enc[sel].tolist()))
                Xl = scaler.transform(X[sel]).astype(np.float32)
                yl = y_enc[sel]
                lv, lt = _split(len(Xl), r2, frac=0.2, floor=16)

                res = {}
                lctx = _ctx(Xl[lt], yl[lt], Xl[lv], yl[lv], n_classes, Xtr.shape[1])
                res["local_only"] = _macro(
                    yev, _predict(_train(lctx), Xev, n_classes).argmax(1), present)
                res["finetune"] = _macro(
                    yev, _predict(_finetune(gmodels, lctx), Xev, n_classes).argmax(1),
                    present)
                for kk in ("Xtr_t", "ytr_t", "Xval_t"):
                    lctx.pop(kk, None)
                torch.cuda.empty_cache()

                Xp = np.vstack([Xtr, Xl])
                yp = np.concatenate([ytr, yl])
                pv, pt = _split(len(Xp), r2)
                pctx = _ctx(Xp[pt], yp[pt], Xp[pv], yp[pv], n_classes, Xtr.shape[1])
                res["pooled"] = _macro(
                    yev, _predict(_train(pctx), Xev, n_classes).argmax(1), present)
                for kk in ("Xtr_t", "ytr_t", "Xval_t"):
                    pctx.pop(kk, None)
                torch.cuda.empty_cache()

                print(f"  {budget:>3} cells (n={len(sel):>6,} rows, "
                      f"{n_cls_local}/{len(present)} classes) rep{rep}: " + "  ".join(
                          f"{a} {v:.4f} ({v - f1_global:+.4f})" for a, v in res.items()))
                for arm, v in res.items():
                    rows.append({"fold": k, "budget": budget, "rep": rep, "arm": arm,
                                 "f1": round(v, 4), "n_local": int(len(sel)),
                                 "n_cells": int(budget),
                                 "n_classes_local": n_cls_local,
                                 "n_classes_eval": len(present),
                                 "delta_vs_global": round(v - f1_global, 4),
                                 "n_eval": int(len(ev))})
        del gmodels
        torch.cuda.empty_cache()

    # aggregate: mean over folds and reps, paired against that fold's global
    agg, nloc = {}, {}
    for r in rows:
        if r["arm"] == "global":
            continue
        agg.setdefault((r["arm"], r["budget"]), []).append(r["delta_vs_global"])
        nloc.setdefault(r["budget"], []).append(r["n_local"])
    summary = [{"arm": a, "budget": b, "n": len(v),
                "n_local_mean": int(np.mean(nloc[b])),
                "delta_vs_global": round(float(np.mean(v)), 4),
                "sd": round(float(np.std(v)), 4)}
               for (a, b), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1]))]
    out = {"budgets": BUDGETS, "n_reps": N_REPS, "n_seeds": N_SEEDS,
           "global_f1_per_fold": [r["f1"] for r in rows if r["arm"] == "global"],
           "rows": rows, "summary": summary,
           "runtime_s": round(time.perf_counter() - t0, 1),
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {OUT}  ({out['runtime_s']}s)")
    print(f"{'arm':12s}{'budget':>9s}{'Δ vs global':>13s}{'sd':>8s}")
    for s in summary:
        print(f"{s['arm']:12s}{s['budget']:>9,}{s['delta_vs_global']:>13.4f}"
              f"{s['sd']:>8.4f}")


if __name__ == "__main__":
    main()
