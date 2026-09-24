"""Unit tests for trials3_arms.py on synthetic data. No GPU, no training.

Run:  ~/myprojects/recover/.venv/bin/python DNN/ar3/test_arms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # DNN/ for data_utils etc

import trials3_arms as TA  # noqa: E402

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL  {msg}")
    else:
        print(f"  OK    {msg}")


def rand_probs(rng, n, c):
    z = rng.random((n, c)) + 0.05
    return z / z.sum(1, keepdims=True)


def is_dist(P):
    return bool(np.all(P >= -1e-9) and np.allclose(P.sum(1), 1.0, atol=1e-6))


class _Trial:
    def __init__(self, config):
        self.config = config


# ---------------------------------------------------------------- tmp_joint
def test_tmp_joint():
    print("\n== tmp_joint ==")
    rng = np.random.default_rng(0)
    nc = 4
    n_locs = 40
    locs, years = [], []
    for l in range(n_locs):
        locs += [l, l]
        years += [2018, 2024]
    meta = pd.DataFrame({"loc": locs, "year": years})
    P = rand_probs(rng, len(locs), nc)

    ctx = {"n_classes": nc, "te_meta": meta, "trial": _Trial({"eps": 0.01})}
    P_out, info = TA.post_tmp_joint(P.copy(), ctx)
    check(is_dist(P_out), "tmp_joint(eps=0.01): valid probability rows")
    check(info["tmp_joint_n_pairs"] == n_locs, "tmp_joint: found all 2018/2024 pairs")
    check("tmp_joint_frac_argmax_changed" in info, "tmp_joint: reports frac_argmax_changed")

    # eps = (C-1)/C -> T uniform -> joint factorises -> marginals == inputs unchanged
    eps_uniform = (nc - 1) / nc
    ctx2 = {"n_classes": nc, "te_meta": meta, "trial": _Trial({"eps": eps_uniform})}
    P_out2, info2 = TA.post_tmp_joint(P.copy(), ctx2)
    check(np.allclose(P_out2, P, atol=1e-6),
          "tmp_joint: eps=(C-1)/C (uniform T) leaves probs ~unchanged")
    check(info2["tmp_joint_frac_argmax_changed"] == 0.0,
          "tmp_joint: eps=(C-1)/C changes no argmax")


# ------------------------------------------------------------------ tmp_hmm
def test_tmp_hmm():
    print("\n== tmp_hmm ==")
    rng = np.random.default_rng(1)
    nc = 4
    n_locs = 30
    years_all = list(range(2017, 2026))
    locs, years = [], []
    for l in range(n_locs):
        for y in years_all:
            locs.append(l)
            years.append(y)
    meta = pd.DataFrame({"loc": locs, "year": years})
    P = rand_probs(rng, len(locs), nc)

    ctx = {"n_classes": nc, "te_meta": meta, "trial": _Trial({"eps": 0.005})}
    P_out, info = TA.post_tmp_hmm(P.copy(), ctx)
    check(is_dist(P_out), "tmp_hmm(eps=0.005): valid probability rows")
    check(info["tmp_hmm_n_locs"] == n_locs, "tmp_hmm: correct loc count")
    check(P_out.shape == P.shape, "tmp_hmm: output shape matches input")

    # eps = (C-1)/C -> transition uniform -> no info crosses time steps ->
    # posterior at each t collapses to the (already-normalised) emission at t
    eps_uniform = (nc - 1) / nc
    ctx2 = {"n_classes": nc, "te_meta": meta, "trial": _Trial({"eps": eps_uniform})}
    P_out2, _ = TA.post_tmp_hmm(P.copy(), ctx2)
    check(np.allclose(P_out2, P, atol=1e-6),
          "tmp_hmm: eps=(C-1)/C (uniform transition) leaves probs ~unchanged")


# ----------------------------------------------------------------- tmp_pool
def test_tmp_pool():
    print("\n== tmp_pool ==")
    rng = np.random.default_rng(2)
    nc = 4
    locs = [0, 0, 0, 1, 1]
    meta = pd.DataFrame({"loc": locs})
    P = rand_probs(rng, len(locs), nc)

    ctx = {"n_classes": nc, "te_meta": meta}
    P_out, info = TA.post_tmp_pool(P.copy(), ctx)
    check(is_dist(P_out), "tmp_pool: valid probability rows")
    check(info["tmp_pool_n_locs"] == 2, "tmp_pool: correct loc count")
    expect0 = P[:3].mean(0)
    expect1 = P[3:].mean(0)
    check(np.allclose(P_out[0], expect0) and np.allclose(P_out[1], expect0)
          and np.allclose(P_out[2], expect0), "tmp_pool: loc 0 rows == mean over its 3 rows")
    check(np.allclose(P_out[3], expect1) and np.allclose(P_out[4], expect1),
          "tmp_pool: loc 1 rows == mean over its 2 rows")


# ----------------------------------------------------------------- tmp_pair
def test_tmp_pair():
    print("\n== tmp_pair partner selection ==")
    # 3 locs: one has 2018+2024 (should pick each other), one has a sparse
    # record to exercise the "closest to 6" tie-break, one is a singleton.
    rows = [
        # loc 0: full 2018/2024 pair (+ decoys)
        (0, 2017), (0, 2018), (0, 2020), (0, 2024),
        # loc 1: only 2017 and 2025 -> for 2017, distance-6 candidates are
        # {2025: d=8} only one candidate so no ambiguity; use a genuine tie:
        (1, 2021), (1, 2017), (1, 2025),
        # loc 2: singleton -> own embedding
        (2, 2019),
    ]
    n = len(rows)
    loc = [r[0] for r in rows]
    year = [r[1] for r in rows]
    meta = pd.DataFrame({"loc": loc, "year": year})
    # X: 64 embedding cols + 3 lidar cols = 67; embedding = row index broadcast
    # so we can identify which row's embedding ended up as the partner.
    X = np.zeros((n, 67), dtype=np.float32)
    for i in range(n):
        X[i, :64] = float(i)
        X[i, 64:] = [100.0 + i, 200.0 + i, 300.0 + i]  # lidar, must pass through untouched

    out = TA.feat_tmp_pair(X, meta, feat_cols=[f"A{i:02d}" for i in range(64)] + ["elevation", "tri", "tch"])
    check(out.shape == (n, 67 + 64), f"tmp_pair: output has 67+64=131 cols, got {out.shape}")
    check(np.array_equal(out[:, :67], X), "tmp_pair: first 67 cols identical to input X")

    def emb_row_idx(row):
        return int(round(out[row, 67]))  # partner embedding value == source row index

    idx_2018 = year.index(2018)  # global row 1 (loc 0)
    idx_2024 = year.index(2024)  # global row 3 (loc 0)
    check(emb_row_idx(idx_2018) == idx_2024, "tmp_pair: 2018 row picks the 2024 row as partner")
    check(emb_row_idx(idx_2024) == idx_2018, "tmp_pair: 2024 row picks the 2018 row as partner")

    # loc 2 singleton -> own embedding
    idx_singleton = loc.index(2)
    check(emb_row_idx(idx_singleton) == idx_singleton, "tmp_pair: singleton loc keeps own embedding")

    # loc 1: years {2021, 2017, 2025}. For row year=2021: others={2017(d=4),
    # 2025(d=4)}, both |d-6|=2 -> tie -> later year (2025) wins.
    rows_loc1 = [i for i, l in enumerate(loc) if l == 1]
    idx_2021 = [i for i in rows_loc1 if year[i] == 2021][0]
    idx_2025 = [i for i in rows_loc1 if year[i] == 2025][0]
    check(emb_row_idx(idx_2021) == idx_2025,
          "tmp_pair: tie-break (equal |d-6|) picks the LATER year")


# ------------------------------------------------------------- tmp_ctx cols
def test_tmp_ctx_first_cols():
    print("\n== tmp_ctx / tmp_ctx_mean: first-67-cols invariance ==")
    rng = np.random.default_rng(3)
    n = 20
    loc = rng.integers(0, 5, size=n)
    year = rng.choice(list(range(2017, 2026)), size=n)
    meta = pd.DataFrame({"loc": loc, "year": year})
    X = rng.normal(size=(n, 67)).astype(np.float32)
    feat_cols = [f"A{i:02d}" for i in range(64)] + ["elevation", "tri", "tch"]

    out_ctx = TA.feat_tmp_ctx(X, meta, feat_cols)
    check(out_ctx.shape == (n, 67 + 128), f"tmp_ctx: shape 67+128, got {out_ctx.shape}")
    check(np.array_equal(out_ctx[:, :67], X), "tmp_ctx: first 67 cols identical to input X")

    out_mean = TA.feat_tmp_ctx_mean(X, meta, feat_cols)
    check(out_mean.shape == (n, 67 + 64), f"tmp_ctx_mean: shape 67+64, got {out_mean.shape}")
    check(np.array_equal(out_mean[:, :67], X), "tmp_ctx_mean: first 67 cols identical to input X")
    check(np.allclose(out_ctx[:, 67:67 + 64], out_mean[:, 67:]),
          "tmp_ctx and tmp_ctx_mean agree on the mean block")

    # sanity: a loc appearing once has std == 0 (ddof=0 population std of 1 pt)
    single_loc = pd.Series(loc).value_counts()
    single = single_loc[single_loc == 1].index
    if len(single):
        rows = np.flatnonzero(loc == single[0])
        std_block = out_ctx[rows[0], 67 + 64:]
        check(np.allclose(std_block, 0.0), "tmp_ctx: singleton loc has std == 0")


def main():
    test_tmp_joint()
    test_tmp_hmm()
    test_tmp_pool()
    test_tmp_pair()
    test_tmp_ctx_first_cols()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
