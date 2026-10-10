# Regenerate the ML-attack tables (III, IV, V) and quality metrics (Table I)
# under the adopted independent-generator model (puf_core v3.0, G4), so the
# manuscript matches the device model described in Section 4.2.
import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch, torch.nn as nn

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, HERE + "/model"); sys.path.insert(0, HERE + "/generation")
import puf_core as P
from generate_c1_c2_datasets import generate_c1_device, generate_c2_device

torch.set_num_threads(2); torch.manual_seed(0)
CH = P.challenge_sequence()
IDX = np.arange(100_000) % 65535
BITS100 = ((CH[IDX][:, None] >> np.arange(16)) & 1).astype(np.float32)
SEEDS = [42, 123, 456]


def c0_pool(ndev, per=100_000):
    X = np.vstack([BITS100] * ndev)
    y = np.concatenate([P.response_vector(CH, P.build_device(d))[IDX] for d in range(ndev)]).astype(np.int64)
    return X, y


def c1_pool(ndev, per=100_000):
    arr = [generate_c1_device(d, per) for d in range(ndev)]
    X = np.vstack([a[:, :16] for a in arr]).astype(np.float32)
    y = np.concatenate([a[:, 16] for a in arr]).astype(np.int64)
    return X, y


def c2_pool(ndev, per=100_000):
    arr = [generate_c2_device(d, per) for d in range(ndev)]
    X = np.vstack([a[:, :16] for a in arr]).astype(np.float32)
    y = np.concatenate([a[:, 16] for a in arr]).astype(np.int64)
    return X, y


class DNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(16, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                              nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def dnn_acc(Xtr, ytr, Xte, yte, opt_name, seed):
    torch.manual_seed(seed)
    m = DNN()
    opt = {"adam": torch.optim.Adam, "nadam": torch.optim.NAdam,
           "rmsprop": torch.optim.RMSprop, "adadelta": torch.optim.Adadelta}[opt_name]
    lr = 1.0 if opt_name == "adadelta" else 1e-3
    o = opt(m.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte)
    bs, ep = 2048, 3
    for e in range(ep):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), bs):
            j = perm[i:i + bs]
            o.zero_grad(); loss = lossf(m(Xtr[j]), ytr[j]); loss.backward(); o.step()
    m.eval()
    with torch.no_grad():
        pred = []
        for i in range(0, len(Xte), 16384):
            pred.append((torch.sigmoid(m(Xte[i:i + 16384])) > 0.5).int().numpy())
    return 100.0 * np.mean(np.concatenate(pred) == yte)


def allmodels(X, y, which, seeds=SEEDS, svm_cap=40000):
    """Return dict model->(mean,std) over seeds for the requested model list."""
    out = {m: [] for m in which}
    for sd in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=sd, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        for m in which:
            if m == "LR":
                a = 100 * LogisticRegression(max_iter=500).fit(Xtr_s, ytr).score(Xte_s, yte)
            elif m == "SVM":
                idx = np.random.default_rng(sd).choice(len(ytr), min(svm_cap, len(ytr)), replace=False)
                mdl = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr_s[idx], ytr[idx])
                a = 100 * mdl.score(Xte_s[:30000], yte[:30000])
            elif m == "MLP":
                a = 100 * MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60,
                                        early_stopping=True, random_state=sd).fit(Xtr_s, ytr).score(Xte_s, yte)
            elif m == "XGB":
                mdl = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                        tree_method="hist", n_jobs=2, verbosity=0, random_state=sd).fit(Xtr, ytr)
                a = 100 * np.mean(mdl.predict(Xte) == yte)
            else:  # DNN/<opt>
                a = dnn_acc(Xtr, ytr, Xte, yte, m.split("/")[1], sd)
            out[m].append(float(a))
    return {m: [float(np.mean(v)), float(np.std(v))] for m, v in out.items()}


res = {}
t0 = time.time()
EIGHT = ["LR", "SVM", "MLP", "DNN/adam", "DNN/nadam", "DNN/rmsprop", "DNN/adadelta", "XGB"]

# ---- Table III: C0, all 8 models, 100K (single device) and 1M (10 devices)
print("Table III C0 100K ...", flush=True)
X, y = c0_pool(1); res["III_C0_100K"] = allmodels(X, y, EIGHT)
print("  ", {k: round(v[0], 2) for k, v in res["III_C0_100K"].items()}, "%.0fs" % (time.time() - t0), flush=True)
print("Table III C0 1M ...", flush=True)
X, y = c0_pool(10); res["III_C0_1M"] = allmodels(X, y, EIGHT)
print("  ", {k: round(v[0], 2) for k, v in res["III_C0_1M"].items()}, "%.0fs" % (time.time() - t0), flush=True)

# ---- Table IV: C0/C1/C2 key models, 100K and 1M
KEY = ["LR", "DNN/adam", "DNN/nadam", "XGB"]
for cfg, fn in [("C0", c0_pool), ("C1", c1_pool), ("C2", c2_pool)]:
    for sz, nd in [("100K", 1), ("1M", 10)]:
        X, y = fn(nd); res["IV_%s_%s" % (cfg, sz)] = allmodels(X, y, KEY)
        print("IV", cfg, sz, {k: round(v[0], 2) for k, v in res["IV_%s_%s" % (cfg, sz)].items()},
              "%.0fs" % (time.time() - t0), flush=True)

json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_regen_g4_results.json"), "w"), indent=1)
print("DONE %.0fs" % (time.time() - t0))
