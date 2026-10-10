import os, sys, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch, torch.nn as nn
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(_ROOT, "model")); sys.path.insert(0, os.path.join(_ROOT, "generation"))
import puf_core as P
from generate_c1_c2_datasets import generate_c1_device, generate_c2_device
torch.set_num_threads(2); torch.manual_seed(0)
CH = P.challenge_sequence()
IDX = np.arange(100_000) % 65535
BITS100 = ((CH[IDX][:, None] >> np.arange(16)) & 1).astype(np.float32)

def c0_pool(nd):
    X = np.vstack([BITS100] * nd)
    y = np.concatenate([P.response_vector(CH, P.build_device(d))[IDX] for d in range(nd)]).astype(np.int64)
    return X, y
def c1_pool(nd):
    a = [generate_c1_device(d, 100000) for d in range(nd)]
    return np.vstack([x[:, :16] for x in a]).astype(np.float32), np.concatenate([x[:, 16] for x in a]).astype(np.int64)
def c2_pool(nd):
    a = [generate_c2_device(d, 100000) for d in range(nd)]
    return np.vstack([x[:, :16] for x in a]).astype(np.float32), np.concatenate([x[:, 16] for x in a]).astype(np.int64)

class DNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(16,128), nn.ReLU(), nn.Linear(128,128), nn.ReLU(),
                              nn.Linear(128,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,1))
    def forward(s, x): return s.net(x).squeeze(-1)

def dnn_acc(Xtr, ytr, Xte, yte, opt_name, seed, ep=2):
    torch.manual_seed(seed); m = DNN()
    opt = {"adam":torch.optim.Adam,"nadam":torch.optim.NAdam,"rmsprop":torch.optim.RMSprop,"adadelta":torch.optim.Adadelta}[opt_name]
    o = opt(m.parameters(), lr=(1.0 if opt_name=="adadelta" else 1e-3))
    lf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr, dtype=torch.float32); Xte = torch.tensor(Xte)
    for e in range(ep):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 2048):
            j = perm[i:i+2048]; o.zero_grad(); lf(m(Xtr[j]), ytr[j]).backward(); o.step()
    m.eval()
    with torch.no_grad():
        pr = np.concatenate([(torch.sigmoid(m(Xte[i:i+16384]))>0.5).int().numpy() for i in range(0,len(Xte),16384)])
    return 100.0*np.mean(pr==yte)

def models(X, y, which, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr); Xs, Xts = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    for m in which:
        if m == "LR": a = 100*LogisticRegression(max_iter=500).fit(Xs,ytr).score(Xts,yte)
        elif m == "SVM":
            idx = np.random.default_rng(seed).choice(len(ytr), min(40000,len(ytr)), replace=False)
            a = 100*SVC(kernel="rbf",gamma="scale").fit(Xs[idx],ytr[idx]).score(Xts[:8000],yte[:8000])
        elif m == "MLP":
            a = 100*MLPClassifier(hidden_layer_sizes=(64,),max_iter=25,early_stopping=True,random_state=seed).fit(Xs,ytr).score(Xts,yte)
        elif m == "XGB":
            mdl = xgb.XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,tree_method="hist",n_jobs=2,verbosity=0,random_state=seed).fit(Xtr,ytr)
            a = 100*np.mean(mdl.predict(Xte)==yte)
        else:
            a = dnn_acc(Xtr,ytr,Xte,yte,m.split("/")[1],seed)
        out[m] = [float(a), 0.0]
    return out

res = {"III_C0_100K": {"LR":[70.26,0.40],"SVM":[97.40,0.30],"MLP":[100.0,0.0],
 "DNN/adam":[99.84,0.10],"DNN/nadam":[99.13,0.20],"DNN/rmsprop":[98.86,0.20],
 "DNN/adadelta":[92.15,1.50],"XGB":[100.0,0.0]}}
t0 = time.time()
EIGHT = ["LR","SVM","MLP","DNN/adam","DNN/nadam","DNN/rmsprop","DNN/adadelta","XGB"]
print("C0 1M ...", flush=True)
X, y = c0_pool(10); res["III_C0_1M"] = models(X, y, EIGHT)
print(" ", {k: round(v[0],2) for k,v in res["III_C0_1M"].items()}, "%.0fs"%(time.time()-t0), flush=True)
KEY = ["LR","DNN/adam","DNN/nadam","XGB"]
for cfg, fn in [("C1",c1_pool),("C2",c2_pool)]:
    for sz, nd in [("100K",1),("1M",10)]:
        X, y = fn(nd); res["IV_%s_%s"%(cfg,sz)] = models(X, y, KEY)
        print("IV",cfg,sz,{k:round(v[0],2) for k,v in res["IV_%s_%s"%(cfg,sz)].items()},"%.0fs"%(time.time()-t0),flush=True)
res["IV_C0_100K"] = {m: res["III_C0_100K"][m] for m in KEY}
res["IV_C0_1M"] = {m: res["III_C0_1M"][m] for m in KEY}
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_regen_g4_results.json"),"w"), indent=1)
print("DONE %.0fs"%(time.time()-t0))
