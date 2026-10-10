# ============================================================
# physical_robustness_study.py
#
# Answers Reviewer 3 and the Handling Editor (second round):
# the ML results rest on simulated CRPs, and real ring
# oscillators behave differently from any simple model.
#
# Physical CRPs could not be collected in this work (no FPGA
# board or measurement equipment was available). This script
# therefore asks a narrower question that CAN be answered
# without hardware: do the two main conclusions of the paper
#   (1) C0 is fully predictable by memorisation on one device,
#   (2) C1 (NLFSR-Shield) reduces prediction to chance,
# survive when the simulated physics is changed in the ways in
# which real silicon is known to differ from our model?
#
# Physical effects modelled (each relative to the reference
# model of puf_core.py v3.0, which is scenario S0):
#
#   Measurement noise   Each evaluation adds Gaussian noise to the
#                       frequency difference of the selected pair,
#                       so bits with a small margin flip most often
#                       (as observed on real RO PUFs). The noise
#                       level is calibrated to a mean bit-flip rate
#                       of 2%, 5% or 10% (reliability 98/95/90%).
#   Layout asymmetry    A fixed delay offset, identical on every
#                       chip, makes every even-numbered oscillator
#                       faster than its odd partner (routing bias).
#   Shared variation    Part of the process variation is common to
#                       all chips (die-to-die / systematic), giving
#                       a correlation rho between the parameters of
#                       any two devices: rho = 0.5 and 0.9.
#   Worst case          rho = 0.9 + layout asymmetry + 5% noise.
#
# For every scenario: 10 devices, 100,000 CRPs per device
# (LFSR seed 0xACE1), C0 and C1 generated from the same noisy
# raw evaluations; XGBoost (200 trees, depth 6), random 70/15/15
# split, 3 seeds. Also reported: reliability, uniformity,
# uniqueness, the bound of the pooled data, prediction of an
# unseen device from the other 9, and how distinct the 32-bit
# NLFSR seeds of different devices are.
#
#     python experiments/physical_robustness_study.py
# Output: experiments/physical_robustness_results.json
# Runtime: about 25 minutes on a 2-core laptop.
# ============================================================
import os, sys, json, time
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import puf_core as P

N_DEV, PER_DEV, SEEDS = 10, 100_000, [42, 123, 456]
CH_ALL = P.challenge_sequence()
IDX = np.arange(PER_DEV) % 65535
CH = CH_ALL[IDX]                                    # challenge of every evaluation, in order
BITS = ((CH_ALL[:, None] >> np.arange(16)) & 1).astype(np.float32)
SHARED_ROOT = 0x5EED20260BADF0F0                    # stream of the shared (systematic) component


def float_device(d):
    """Unrounded Gaussian parameters of device d (same streams as puf_core)."""
    g = np.random.default_rng(np.random.SeedSequence(P.DEVICE_ROOT_SEED, spawn_key=(d,)))
    return g.normal(0, P.BASE_SIGMA, P.NUM_ROS), g.normal(0, P.STAGE_SIGMA, (P.NUM_ROS, P.STAGES, 2))


def shared_component():
    g = np.random.default_rng(np.random.SeedSequence(SHARED_ROOT))
    return g.normal(0, P.BASE_SIGMA, P.NUM_ROS), g.normal(0, P.STAGE_SIGMA, (P.NUM_ROS, P.STAGES, 2))


def margins(rho=0.0, asym=0.0):
    """Frequency difference f(a) - f(b) for all 65,535 challenges, per device."""
    sb, sd = shared_component()
    out = []
    for d in range(N_DEV):
        ib, idl = float_device(d)
        base = np.sqrt(rho) * sb + np.sqrt(1 - rho) * ib
        delta = np.sqrt(rho) * sd + np.sqrt(1 - rho) * idl
        base = base + asym * P.BASE_SIGMA * (np.arange(P.NUM_ROS) % 2 == 0)   # even ROs faster
        dev = (np.rint(base).astype(np.int64), np.rint(delta).astype(np.int64))
        f = np.repeat(dev[0][None, :], 65535, 0).astype(float)
        for ro in range(P.NUM_ROS):
            for s in range(P.STAGES):
                bb = (CH_ALL >> ((ro + 5 * s) & 15)) & 1
                f[:, ro] += np.where(bb == 1, dev[1][ro, s, 1], dev[1][ro, s, 0])
        a = CH_ALL & 0xF
        r = np.arange(65535)
        out.append(f[r, a] - f[r, a ^ 1])
    return np.array(out)                                 # (N_DEV, 65535)


def noise_sigma_for(M, target):
    """Noise std giving mean flip probability `target` (bisection)."""
    from scipy.stats import norm
    if target == 0:
        return 0.0
    lo, hi = 1e-3, 1e5
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        pf = np.mean(norm.cdf(-np.abs(M) / mid))
        lo, hi = (mid, hi) if pf < target else (lo, mid)
    return np.sqrt(lo * hi)


def nlfsr_stream(seed_bits, n):
    s = 0
    for b in seed_bits:
        s = ((s << 1) | int(b)) & 0xFFFFFFFF
    seed = s
    out = np.empty(n, np.uint8)
    for t in range(n):
        lin = ((s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s) & 1
        a, b, c = (s >> 30) & 1, (s >> 15) & 1, (s >> 7) & 1
        k = lin ^ ((a & b) ^ (b & c) ^ (a & c))
        s = ((s << 1) | k) & 0xFFFFFFFF
        out[t] = k
    return seed, out


def xgb_acc(X, y, seeds=SEEDS):
    v = []
    for sd in seeds:
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=sd, stratify=y)
        _, Xte, _, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=sd, stratify=ytmp)
        m = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                              random_state=sd, n_jobs=2, verbosity=0).fit(Xtr, ytr)
        v.append(100 * np.mean(m.predict(Xte) == yte))
    return [float(np.mean(v)), float(np.std(v))]


def pooled_bound(labels):
    """Highest accuracy of any challenge-only predictor on the pooled rows."""
    cid = np.tile(IDX, N_DEV)
    ones = np.bincount(cid, weights=labels, minlength=65535)
    cnt = np.bincount(cid, minlength=65535)
    return float(100 * np.sum(np.maximum(ones, cnt - ones)) / cnt.sum())


SCEN = [("S0", "reference model (puf_core v3.0)",            dict(rho=0.0, asym=0.0, noise=0.00)),
        ("N2", "measurement noise, reliability 98%",          dict(rho=0.0, asym=0.0, noise=0.02)),
        ("N5", "measurement noise, reliability 95%",          dict(rho=0.0, asym=0.0, noise=0.05)),
        ("N10", "measurement noise, reliability 90%",         dict(rho=0.0, asym=0.0, noise=0.10)),
        ("A", "layout asymmetry common to all chips",          dict(rho=0.0, asym=1.0, noise=0.00)),
        ("R5", "shared process variation, rho = 0.5",         dict(rho=0.5, asym=0.0, noise=0.00)),
        ("R9", "shared process variation, rho = 0.9",         dict(rho=0.9, asym=0.0, noise=0.00)),
        ("W", "worst case: rho 0.9 + asymmetry + 5% noise",   dict(rho=0.9, asym=1.0, noise=0.05))]

res = {}
t_all = time.time()
for si, (sid, sname, cfg) in enumerate(SCEN):
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence(2026, spawn_key=(si,)))
    M = margins(cfg["rho"], cfg["asym"])                   # (10, 65535)
    sig = noise_sigma_for(M, cfg["noise"])
    noiseless = (M > 0).astype(np.uint8)
    raw = np.empty((N_DEV, PER_DEV), np.uint8)             # raw bit of every evaluation
    for d in range(N_DEV):
        m = M[d][IDX]
        raw[d] = ((m + (rng.normal(0, sig, PER_DEV) if sig > 0 else 0)) > 0).astype(np.uint8)
    flip = float(np.mean(raw != noiseless[:, IDX]))

    # C1: first 32 raw evaluations seed the NLFSR, then z = r XOR k
    z = np.empty((N_DEV, PER_DEV), np.uint8)
    seeds = []
    for d in range(N_DEV):
        rr = np.concatenate([raw[d], ((M[d][(np.arange(PER_DEV, PER_DEV + 32)) % 65535] +
                              (rng.normal(0, sig, 32) if sig > 0 else 0)) > 0).astype(np.uint8)])
        seed, k = nlfsr_stream(rr[:32], PER_DEV)
        seeds.append(seed)
        # challenges for the output phase start after the 32 seeding evaluations
        z[d] = rr[32:32 + PER_DEV] ^ k
    # C1 output phase uses challenges 32 .. 32+PER_DEV-1
    IDX1 = (np.arange(PER_DEV) + 32) % 65535

    seed_arr = np.array(seeds, dtype=np.uint64)
    hd = [bin(int(seed_arr[i]) ^ int(seed_arr[j])).count("1") for i in range(N_DEV) for j in range(i + 1, N_DEV)]
    # seed reproduction: probability a device re-derives its own seed at a later power-up
    if sig > 0:
        from scipy.stats import norm
        m32 = M[:, :32]
        p_ok = norm.cdf(np.abs(m32) / sig)                 # per-bit probability of the noiseless value
        seed_repro = float(np.mean(np.prod(p_ok, axis=1)))
    else:
        seed_repro = 1.0

    r = dict(name=sname, **cfg,
             reliability=100 * (1 - flip),
             uniformity=float(100 * noiseless.mean()),
             uniqueness=float(100 * np.mean([np.mean(noiseless[i] != noiseless[j])
                                             for i in range(N_DEV) for j in range(i + 1, N_DEV)])),
             seed_identical_pairs=int(sum(h == 0 for h in hd)), seed_pairs=len(hd),
             seed_min_hd=int(min(hd)), seed_mean_hd=float(np.mean(hd)),
             seed_reproduction=100 * seed_repro)

    # unseen-device prediction (C0, noiseless tables): majority of the other 9 devices
    lodo = []
    for d in range(N_DEV):
        k1 = np.delete(noiseless, d, 0).sum(0)
        pred = np.where(k1 * 2 == N_DEV - 1, 1, (k1 * 2 > N_DEV - 1)).astype(np.uint8)
        lodo.append(100 * np.mean(pred == noiseless[d]))
    r["unseen_device_c0"] = [float(np.mean(lodo)), float(np.std(lodo, ddof=1))]

    # single device (device 0)
    perm = np.random.default_rng(42).permutation(PER_DEV)
    r["c0_single"] = xgb_acc(BITS[IDX][perm], raw[0][perm])
    r["c1_single"] = xgb_acc(BITS[IDX1][perm], z[0][perm])

    # pooled 10 devices
    permP = np.random.default_rng(42).permutation(N_DEV * PER_DEV)
    Xp0 = np.vstack([BITS[IDX]] * N_DEV)[permP]
    Xp1 = np.vstack([BITS[IDX1]] * N_DEV)[permP]
    y0, y1 = raw.reshape(-1), z.reshape(-1)
    r["c0_pooled"] = xgb_acc(Xp0, y0[permP])
    r["c1_pooled"] = xgb_acc(Xp1, y1[permP])
    r["c0_pooled_bound"] = pooled_bound(y0)
    cid1 = np.tile(IDX1, N_DEV)
    ones = np.bincount(cid1, weights=y1, minlength=65535); cnt = np.bincount(cid1, minlength=65535)
    r["c1_pooled_bound"] = float(100 * np.sum(np.maximum(ones, cnt - ones)) / cnt.sum())

    res[sid] = r
    print("\n%-4s %s  (%.0f s)" % (sid, sname, time.time() - t0))
    print("  reliability %.2f%%  uniformity %.2f%%  uniqueness %.2f%%  unseen device (C0) %.2f +- %.2f%%"
          % (r["reliability"], r["uniformity"], r["uniqueness"], *r["unseen_device_c0"]))
    print("  C0 single %.2f +- %.2f   C1 single %.2f +- %.2f" % (*r["c0_single"], *r["c1_single"]))
    print("  C0 pooled %.2f +- %.2f (bound %.2f)   C1 pooled %.2f +- %.2f (bound %.2f)"
          % (*r["c0_pooled"], r["c0_pooled_bound"], *r["c1_pooled"], r["c1_pooled_bound"]))
    print("  NLFSR seeds: identical pairs %d of %d, min HD %d, mean HD %.1f, seed reproduction %.2f%%"
          % (r["seed_identical_pairs"], r["seed_pairs"], r["seed_min_hd"], r["seed_mean_hd"],
             r["seed_reproduction"]), flush=True)

json.dump(res, open(os.path.join(HERE, "physical_robustness_results.json"), "w"), indent=1)
print("\nDone in %.0f s" % (time.time() - t_all))
