# ============================================================
# independent_seed_study.py
#
# Answers Reviewer 1 (comment 1) and the Handling Editor:
# repeat the multi-device experiment with independently
# generated, non-chained device seeds, and separate effects of
# the PUF construction from effects of the LCG-based simulation.
#
# Four device generators are compared. The PUF model (three CIU
# stages per oscillator, pair a = C[3:0], b = a XOR 1) and the
# challenge sequence (16-bit LFSR, seed 0xACE1) are identical in
# all four; only the way the device parameters are drawn changes.
#
#   G1  chained 16-bit LCG seeds (versions 1 and 2 of the paper)
#   G2  independent random 16-bit seeds, same LCG, with seeds
#       rejected if their 112-draw windows overlap on the LCG cycle
#   G3  independent PCG64 stream per device, same uniform windows
#   G4  independent PCG64 stream per device, Gaussian variation
#       with the same standard deviation  (= puf_core v3.0)
#
# Part A  generator diagnostics on 30 devices
# Part B  pooled 1M CRPs from 10 devices, protocol of Table III
#         (random 70/15/15 split, 5 seeds), LR and XGBoost, and
#         the Bayes ceiling of any challenge-only predictor
# Part C  30-device leave-one-device-out transfer: can responses
#         of 29 devices predict an unseen 30th device?
# Part D  single device, 100K CRPs (device 0 of each generator)
#
# Run from the package root:
#     python experiments/independent_seed_study.py
# Output: experiments/independent_seed_study_results.json
# Runtime: about 20 minutes on a 2-core laptop.
# ============================================================
import os, sys, json, time
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import puf_core as P
import legacy_lcg as L

SEEDS = [42, 123, 456, 789, 1234]
N_POOL, N_ALL, PER_DEV = 10, 30, 100_000
G3_ROOT = 0x5EED20260BADF00D          # G4 uses P.DEVICE_ROOT_SEED (= G3_ROOT + 1)

CH = P.challenge_sequence()                                   # 65,535 challenges
BITS = ((CH[:, None] >> np.arange(16)) & 1).astype(np.float32)


# ---------------- the four generators ----------------
def g1_chained_lcg(n):
    return [L.build_device_lcg(s) for s in L.chained_seeds(n)]


_POS = L.lcg_position_map()
def g2_independent_lcg(n):
    rng = np.random.default_rng(np.random.SeedSequence(G3_ROOT).spawn(1)[0])
    seeds, pos = [], []
    while len(seeds) < n:
        cand = int(rng.integers(0, 65536)); p = _POS[cand]
        if all(p[0] != q[0] or min((p[1] - q[1]) % 32768, (q[1] - p[1]) % 32768) > 112
               for q in pos):
            seeds.append(cand); pos.append(p)
    return [L.build_device_lcg(s) for s in seeds]


def g3_independent_uniform(n):
    out = []
    for child in np.random.SeedSequence(G3_ROOT).spawn(n):
        g = np.random.default_rng(child)
        base = g.integers(0, L.BASE_SPREAD, P.NUM_ROS) - L.BASE_SPREAD // 2
        delta = g.integers(0, L.STAGE_SPREAD, (P.NUM_ROS, P.STAGES, 2)) - L.STAGE_SPREAD // 2
        out.append((base.astype(np.int64), delta.astype(np.int64)))
    return out


def g4_independent_gaussian(n):
    return [P.build_device(d) for d in P.device_ids(n)]


GENERATORS = [("G1", "chained LCG seeds (as published)", g1_chained_lcg),
              ("G2", "independent seeds, same LCG", g2_independent_lcg),
              ("G3", "independent PCG64, uniform", g3_independent_uniform),
              ("G4", "independent PCG64, Gaussian (adopted)", g4_independent_gaussian)]


def xgb_model(seed):
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             tree_method="hist", random_state=seed, n_jobs=2, verbosity=0)


def lr_acc(Xtr, ytr, Xte, yte, w=None):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr, sample_weight=w)
    return 100 * m.score(sc.transform(Xte), yte)


def pooled_dataset(tables):
    idx = np.arange(PER_DEV) % 65535
    X = np.vstack([BITS[idx]] * len(tables))
    y = np.concatenate([t[idx] for t in tables]).astype(np.int32)
    perm = np.random.default_rng(42).permutation(len(y))     # shuffle as in the paper
    return X[perm], y[perm]


def split(X, y, seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=seed, stratify=ytmp)
    return Xtr, ytr, Xte, yte


def ms(v):
    return [float(np.mean(v)), float(np.std(v))]


results = {}
t_all = time.time()
for gid, gname, gen in GENERATORS:
    t0 = time.time()
    devs = gen(N_ALL)
    T = np.array([P.response_vector(CH, d) for d in devs]).astype(np.int8)   # (30, 65535)
    r = {"name": gname}

    # ---------- Part A: diagnostics ----------
    C = np.corrcoef(T.astype(float))
    r["uniformity"] = 100 * T.mean()
    r["uniqueness"] = 100 * np.mean([np.mean(T[i] != T[j])
                                     for i in range(N_ALL) for j in range(i + 1, N_ALL)])
    r["neighbour_r"] = float(np.mean([C[i, i + 1] for i in range(N_ALL - 1)]))
    r["non_neighbour_r"] = float(np.mean([C[i, j] for i in range(N_ALL)
                                          for j in range(i + 2, N_ALL)]))
    r["effective_bits_per_device"] = float(np.mean(
        [sum(bool(np.any(P.response_vector(CH ^ (1 << p), d) != P.response_vector(CH, d)))
             for p in range(16)) for d in devs]))

    # ---------- Part B: pooled 1M, ten devices ----------
    T10 = T[:N_POOL]
    k = T10.sum(0)
    w = np.where(np.arange(65535) < PER_DEV - 65535, 2, 1)     # challenges seen twice per device
    r["bayes_ceiling"] = float(100 * np.sum(w * np.maximum(k, N_POOL - k)) / (N_POOL * w.sum()))
    X, y = pooled_dataset(T10)
    lr_v, xg_v = [], []
    for sd in SEEDS:
        Xtr, ytr, Xte, yte = split(X, y, sd)
        lr_v.append(lr_acc(Xtr, ytr, Xte, yte))
        xg_v.append(100 * np.mean(xgb_model(sd).fit(Xtr, ytr).predict(Xte) == yte))
    r["pooled_1M"] = {"LR": ms(lr_v), "XGBoost": ms(xg_v)}

    # ---------- Part C: leave-one-device-out over 30 devices ----------
    # Training data = the full response tables of the other 29 devices,
    # given as one weighted row per (challenge, label).
    maj, xgl = [], []
    for d in range(N_ALL):
        others = np.delete(T, d, 0)
        k1 = others.sum(0)
        maj.append(100 * np.mean((k1 > (N_ALL - 1) / 2).astype(np.int8) == T[d]))
        Xw = np.vstack([BITS, BITS]); yw = np.r_[np.ones(65535), np.zeros(65535)]
        ww = np.r_[k1, (N_ALL - 1) - k1].astype(float); keep = ww > 0
        m = xgb_model(42).fit(Xw[keep], yw[keep], sample_weight=ww[keep])
        xgl.append(100 * np.mean(m.predict(BITS) == T[d]))
    for key, v in (("lodo_majority", maj), ("lodo_xgboost", xgl)):
        v = np.array(v); t, pv = stats.ttest_1samp(v, 50.0)
        r[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p_vs_50": float(pv)}

    # ---------- Part D: single device, 100K ----------
    idx = np.arange(PER_DEV) % 65535
    X1, y1 = BITS[idx], T[0][idx].astype(np.int32)
    perm = np.random.default_rng(42).permutation(PER_DEV); X1, y1 = X1[perm], y1[perm]
    lr_v, xg_v = [], []
    for sd in SEEDS:
        Xtr, ytr, Xte, yte = split(X1, y1, sd)
        lr_v.append(lr_acc(Xtr, ytr, Xte, yte))
        xg_v.append(100 * np.mean(xgb_model(sd).fit(Xtr, ytr).predict(Xte) == yte))
    r["single_100K"] = {"LR": ms(lr_v), "XGBoost": ms(xg_v), "uniformity": float(100 * T[0].mean())}

    results[gid] = r
    print("\n%s  %s   (%.0f s)" % (gid, gname, time.time() - t0))
    print("  A  uniformity %.2f%%  uniqueness %.2f%%  neighbour r %+.3f  non-neighbour r %+.3f"
          "  effective bits/device %.2f" % (r["uniformity"], r["uniqueness"], r["neighbour_r"],
                                            r["non_neighbour_r"], r["effective_bits_per_device"]))
    print("  B  pooled 1M (10 dev): LR %.2f +- %.2f   XGB %.2f +- %.2f   Bayes ceiling %.2f%%"
          % (*r["pooled_1M"]["LR"], *r["pooled_1M"]["XGBoost"], r["bayes_ceiling"]))
    print("  C  unseen device (30-fold): majority-of-29 %.2f +- %.2f (p=%.3g)   XGB %.2f +- %.2f (p=%.3g)"
          % (r["lodo_majority"]["mean"], r["lodo_majority"]["sd"], r["lodo_majority"]["p_vs_50"],
             r["lodo_xgboost"]["mean"], r["lodo_xgboost"]["sd"], r["lodo_xgboost"]["p_vs_50"]))
    print("  D  single device 100K: LR %.2f +- %.2f   XGB %.2f +- %.2f   (device-0 uniformity %.2f%%)"
          % (*r["single_100K"]["LR"], *r["single_100K"]["XGBoost"], r["single_100K"]["uniformity"]),
          flush=True)

json.dump(results, open(os.path.join(HERE, "independent_seed_study_results.json"), "w"), indent=1)
print("\nDone in %.0f s" % (time.time() - t_all))
