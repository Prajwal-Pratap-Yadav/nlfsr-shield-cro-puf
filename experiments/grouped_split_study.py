# ============================================================
# grouped_split_study.py
#
# Reviewer 4, comments 3 and 4 (second round):
#   - a challenge-grouped train/test split, so that no challenge
#     appears in both training and test (R4.3);
#   - complete per-device results, separating single-device from
#     pooled multi-device experiments (R4.4).
#
# Ten devices (G4, puf_core v3.0), 100,000 CRPs each. Stateless
# models (the standard PUF modelling-attack threat model):
# Logistic Regression, an MLP, and XGBoost. Three splits:
#   random      70/15/15 row split (as in Section 5);
#   challenge   grouped by the 16-bit challenge (GroupShuffleSplit),
#               so a challenge is wholly in train or wholly in test;
#   class       grouped by the 416 effective challenge classes of
#               the CRO-PUF (Section 4.10), the strongest split:
#               no challenge whose response is determined by a
#               class seen in training can appear in test.
#
# Reported per device and configuration: accuracy of each model
# under each split, so the paper can give full per-device tables
# instead of pooled summaries only.
#
#     python experiments/grouped_split_study.py
# Output: experiments/grouped_split_results.json
# ============================================================
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "generation"))
import puf_core as P
from generate_c1_c2_datasets import generate_c1_device, KeccakF200Model

N_DEV, PER_DEV, SEEDS = 10, 100_000, [42, 123, 456]
CH = P.challenge_sequence()
BITS = ((CH[:, None] >> np.arange(16)) & 1).astype(np.float32)
IDX = np.arange(PER_DEV) % 65535

# 416 effective-class id of every challenge (Section 4.10)
def class_bits(a):
    b = a ^ 1
    return sorted({(a + 5 * s) & 15 for s in range(3)} | {(b + 5 * s) & 15 for s in range(3)})
def eff_key(c):
    a = c & 0xF; k = a
    for p in class_bits(a):
        k = (k << 1) | ((c >> p) & 1)
    return k
CLS = np.array([eff_key(int(c)) for c in CH])


def c0_response(dev_id):
    return P.response_vector(CH, P.build_device(dev_id))

def c1_response(dev_id):
    d = generate_c1_device(dev_id, PER_DEV)       # ordered C1 stream, 100k
    return d[:, :16].astype(np.float32), d[:, 16].astype(np.int32)

def models():
    return {
        "LR":  lambda: LogisticRegression(max_iter=500),
        "MLP": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=80,
                                     early_stopping=True, n_iter_no_change=6, random_state=0),
        "XGB": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         tree_method="hist", n_jobs=2, verbosity=0, random_state=0),
    }

def evaluate(X, y, groups, split, seed):
    if split == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.15, random_state=seed, stratify=y)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        tr, te = next(gss.split(X, y, groups=groups))
    out = {}
    needs_scale = {"LR", "MLP"}
    for name, mk in models().items():
        Xtr, Xte = X[tr], X[te]
        if name in needs_scale:
            sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        m = mk().fit(Xtr, y[tr])
        out[name] = 100.0 * np.mean(m.predict(Xte) == y[te])
    return out

res = {"C0": {}, "C1": {}}
t0 = time.time()
for cfg in ("C0", "C1"):
    for d in range(N_DEV):
        if cfg == "C0":
            X = BITS[IDX]; y = c0_response(d)[IDX].astype(np.int32)
            grp_ch = IDX; grp_cls = CLS[IDX]
            uni = 100.0 * y.mean()
        else:
            X, y = c1_response(d)
            # the C1 output phase starts after the 32 seeding evaluations,
            # so take the challenge of every row from its own bits
            ch_int = (X.astype(np.int64) * (1 << np.arange(16))).sum(1)
            pos = {int(c): i for i, c in enumerate(CH)}
            grp_ch = np.array([pos[int(c)] for c in ch_int])
            grp_cls = CLS[grp_ch]
            uni = 100.0 * y.mean()
        dev = {"uniformity": float(uni)}
        for split, groups in (("random", None), ("challenge", grp_ch), ("class", grp_cls)):
            acc = {m: [] for m in ("LR", "MLP", "XGB")}
            for sd in SEEDS:
                e = evaluate(X, y, groups, split, sd)
                for m in acc: acc[m].append(e[m])
            dev[split] = {m: [float(np.mean(v)), float(np.std(v))] for m, v in acc.items()}
        res[cfg]["dev%d" % d] = dev
        print("%s device %d  unif %.2f%%  | random XGB %.2f  challenge XGB %.2f  class XGB %.2f  | class LR %.2f MLP %.2f"
              % (cfg, d, uni, dev["random"]["XGB"][0], dev["challenge"]["XGB"][0], dev["class"]["XGB"][0],
                 dev["class"]["LR"][0], dev["class"]["MLP"][0]), flush=True)

# summary across devices
for cfg in ("C0", "C1"):
    agg = {}
    for split in ("random", "challenge", "class"):
        for m in ("LR", "MLP", "XGB"):
            vals = [res[cfg]["dev%d" % d][split][m][0] for d in range(N_DEV)]
            agg["%s_%s" % (split, m)] = [float(np.mean(vals)), float(np.std(vals)),
                                         float(np.min(vals)), float(np.max(vals))]
    res[cfg]["summary"] = agg

json.dump(res, open(os.path.join(HERE, "grouped_split_results.json"), "w"), indent=1)
print("\ndone in %.0fs" % (time.time() - t0))
