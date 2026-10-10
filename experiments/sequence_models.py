# ============================================================
# sequence_models.py
#
# Reviewer 4, comment 2 (second round): evaluate sequence-aware
# models (LSTM, GRU) against the stateful C1 construction, since
# the stateless models of Section 5 cannot see the time order.
#
# The C1 output stream is read IN GENERATION ORDER (not shuffled),
# so a recurrent model has access to exactly the information a
# sequence-aware adversary would: the challenge of each step and
# the history of previous outputs. Two input settings per model:
#
#   challenge-only : predict z_t from the 16 challenge bits
#                    (matches the Section 5 threat model);
#   sequence       : predict z_t from the 16 challenge bits plus
#                    a window of the previous W (challenge, output)
#                    pairs, so the model can in principle pick up
#                    any temporal structure in the keystream.
#
# Baselines C0 (unprotected) and C1 (NLFSR-Shield), device 0,
# 100,000 ordered CRPs, 70/30 time-ordered split (train on the
# first 70%, test on the last 30%, so no future leaks into the
# past). LSTM and GRU, one hidden layer of 64 units.
#
#     python experiments/sequence_models.py
# Output: experiments/sequence_models_results.json
# ============================================================
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "generation"))
import puf_core as P
from generate_c1_c2_datasets import generate_c1_device

import torch
import torch.nn as nn
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

N, W = 100_000, 64
CH = P.challenge_sequence()
IDX = np.arange(N) % 65535


def streams():
    # C0: ordered raw responses of device 0
    r0 = P.response_vector(CH, P.build_device(0))[IDX].astype(np.float32)
    x0 = ((CH[IDX][:, None] >> np.arange(16)) & 1).astype(np.float32)
    # C1: ordered obfuscated stream of device 0
    d = generate_c1_device(0, N)
    x1 = d[:, :16].astype(np.float32)
    r1 = d[:, 16].astype(np.float32)
    return {"C0": (x0, r0), "C1": (x1, r1)}


class Rec(nn.Module):
    def __init__(self, kind, in_dim, hid=64):
        super().__init__()
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn(in_dim, hid, batch_first=True)
        self.fc = nn.Linear(hid, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_windows(x, y, seq):
    """Return (samples, W or 1, feat) tensors. seq=True uses the previous
    W (challenge,output) pairs plus the current challenge; seq=False uses
    the current challenge only (as a length-1 sequence)."""
    n = len(y)
    if not seq:
        return x[:, None, :], y
    feat = x.shape[1] + 1
    Xw = np.zeros((n - W, W + 1, feat), np.float32)
    for t in range(W, n):
        # previous W steps: their challenge bits + their output bit
        Xw[t - W, :W, :x.shape[1]] = x[t - W:t]
        Xw[t - W, :W, x.shape[1]] = y[t - W:t]
        # current step: challenge bits only, output unknown (0)
        Xw[t - W, W, :x.shape[1]] = x[t]
    return Xw, y[W:]


def run(kind, X, y, epochs=6):
    ntr = int(0.7 * len(y))                      # time-ordered split
    Xtr = torch.tensor(X[:ntr]); ytr = torch.tensor(y[:ntr])
    Xte = torch.tensor(X[ntr:]); yte = torch.tensor(y[ntr:])
    model = Rec(kind, X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    bs = 512
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[j]), ytr[j]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = 0; nb = 0
        for i in range(0, len(yte), 4096):
            p = (torch.sigmoid(model(Xte[i:i + 4096])) > 0.5).float()
            acc += (p == yte[i:i + 4096]).sum().item(); nb += len(p)
    return 100.0 * acc / nb


res = {}
S = streams()
t0 = time.time()
for cfg in ("C0", "C1"):
    x, y = S[cfg]
    res[cfg] = {}
    for seq in (False, True):
        Xw, yw = make_windows(x, y, seq)
        label = "sequence" if seq else "challenge-only"
        res[cfg][label] = {}
        for kind in ("LSTM", "GRU"):
            a = run(kind, Xw, yw)
            res[cfg][label][kind] = float(a)
            print("%s  %-14s  %-4s  test acc %.2f%%   (%.0fs)"
                  % (cfg, label, kind, a, time.time() - t0), flush=True)

json.dump(res, open(os.path.join(HERE, "sequence_models_results.json"), "w"), indent=1)
print("done in %.0fs" % (time.time() - t0))
