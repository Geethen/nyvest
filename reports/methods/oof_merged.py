"""Re-run arm B of exp_merge_bare_sparse (9-class, moe_shared@8) keeping OOF probs."""
import sys, time, numpy as np
sys.path.insert(0, "/home/geethen.singh/myprojects/nyvest/DNN")
import exp_merge_bare_sparse as E, data_utils as du, dnn_core as C
cfg = C.Config()
print("arch", cfg.arch, flush=True)
du.MERGE_EXTRA = "11:2"
data = C.load_cached("lidar")
X, y, g = data["X"], data["y_enc"], data["groups"]
classes = data["classes"]; n = len(classes)
cls12 = classes.index(12)
C.set_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
P = np.zeros((len(y), n), np.float32); fold = np.full(len(y), -1)
for k, tr, te in du.fold_indices(y, g):
    t = time.perf_counter()
    keep = du.clean_stale_class_mask(X[tr], y[tr], None, cls12, data["lon"][tr], data["lat"][tr])
    ens = C.fit_ensemble(X[tr[keep]], y[tr[keep]], n, cfg, data["feat_cols"], classes, data.get("lidar_med"), rng)
    P[te] = ens.predict_proba(X[te]); fold[te] = k
    print(k, du.macro_f1(y[te], P[te].argmax(1), n), time.perf_counter()-t, flush=True)
print("pooled", du.macro_f1(y, P.argmax(1), n), flush=True)
np.savez_compressed(sys.argv[1], P=P, y=y, fold=fold, classes=np.array(classes),
                    lon=data["lon"], lat=data["lat"], groups=g,
                    year=data.get("year", np.zeros(0)))
