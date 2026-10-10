# ============================================================
# run_nist.py
#
# Runs the NIST SP 800-22 battery, the SP 800-90B non-IID
# min-entropy estimators and the autocorrelation analysis on
# the sequential (non-shuffled) response streams.
#
# Usage:  python run_nist.py
# Output: nist_results.txt
# ============================================================

import math
import numpy as np
import pandas as pd
from nist_sp800_22 import run_battery
from scipy.special import erfc

FILES = {
    "C0_unprotected":  "sim/crp_output/crp_c0_1m_sequential.csv",
    "C1_NLFSR_Shield": "sim/crp_output/crp_c1_1m_sequential.csv",
    "C2_Keccak":       "sim/crp_output/crp_c2_1m_sequential.csv",
}

for _lbl, _p in FILES.items():
    if "sequential" not in _p:
        raise SystemExit("ERROR: %s is not a sequential file." % _p)


def mcv_min_entropy(bits):
    n = len(bits)
    p_hat = max(np.mean(bits == 0), np.mean(bits == 1))
    p_u = min(1.0, p_hat + 2.5758293035489008 * math.sqrt(p_hat * (1 - p_hat) / n))
    return -math.log2(p_u), p_hat, p_u


def markov_min_entropy(bits):
    c0 = int(np.sum(bits[:-1] == 0)); c1 = int(np.sum(bits[:-1] == 1))
    p00 = np.sum((bits[:-1] == 0) & (bits[1:] == 0)) / max(c0, 1)
    p11 = np.sum((bits[:-1] == 1) & (bits[1:] == 1)) / max(c1, 1)
    P = {(0, 0): p00, (0, 1): 1 - p00, (1, 0): 1 - p11, (1, 1): p11}
    p0 = float(np.mean(bits == 0)); best = 0.0
    for start, ps in ((0, p0), (1, 1 - p0)):
        prob, cur = ps, start
        for _ in range(127):
            nxt = 0 if P[(cur, 0)] >= P[(cur, 1)] else 1
            prob *= P[(cur, nxt)]; cur = nxt
            if prob == 0: break
        best = max(best, prob)
    return 1.0 if best == 0 else min(1.0, -math.log2(best) / 128)


def autocorrelation(bits, max_lag=4096):
    x = bits.astype(np.float64) * 2 - 1
    x -= x.mean()
    den = float(np.dot(x, x)); n = len(x)
    r = np.array([float(np.dot(x[:-l], x[l:]) / den) for l in range(1, max_lag + 1)])
    return r, 1.96 / math.sqrt(n)


out = open("nist_results.txt", "w", encoding="utf-8")
def show(s=""):
    print(s); out.write(s + "\n"); out.flush()

for label, path in FILES.items():
    show("=" * 72)
    show("CONFIGURATION: " + label)
    show("source file  : " + path)
    show("=" * 72)

    bits = pd.read_csv(path, usecols=["response"])["response"].astype(np.uint8).values
    show("n = %d bits    ones fraction = %.6f" % (len(bits), bits.mean()))

    show("\n--- NIST SP 800-22 Rev.1a ---")
    run_battery(bits)

    show("\n--- NIST SP 800-90B (non-IID) ---")
    h1, p_hat, p_u = mcv_min_entropy(bits)
    h3 = markov_min_entropy(bits)
    show("  Most Common Value : H_min = %.5f  (p_hat=%.6f)" % (h1, p_hat))
    show("  Markov            : H_min = %.5f" % h3)
    show("  ---> min-entropy  = %.5f bits/bit" % min(h1, h3))

    show("\n--- Autocorrelation, lags 1..4096 ---")
    r, bound = autocorrelation(bits)
    show("  95%% bound                : +/- %.6f" % bound)
    show("  max |r|                  : %.6f at lag %d"
         % (np.abs(r).max(), int(np.argmax(np.abs(r))) + 1))
    show("  lags exceeding the bound : %d of %d (expect ~205 by chance)"
         % (int((np.abs(r) > bound).sum()), len(r)))
    show("")

out.close()
print("\nAll done. Results saved to nist_results.txt")
