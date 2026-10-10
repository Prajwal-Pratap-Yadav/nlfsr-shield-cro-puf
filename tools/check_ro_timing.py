# ============================================================
# check_ro_timing.py
#
# Compares the responses of the timing-simulated ring-oscillator
# RTL (ro_timing_out.txt, from rtl_hw/tb_ro_timing.v) with the
# Python model, and reports where any disagreement occurs in
# terms of the model's frequency margin |f(a) - f(b)|.
#
#   python tools/check_ro_timing.py [device_id]
# ============================================================
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from puf_core import build_device, compute_raw_response, ro_frequency, lfsr_next, LFSR_SEED

dev = build_device(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
rows = [l.split() for l in open("ro_timing_out.txt") if l.strip()]
lf, bad_ch, out = LFSR_SEED, 0, []
for ch_hex, bit, ca, cb in rows:
    ch = int(ch_hex, 16)
    bad_ch += ch != lf
    lf = lfsr_next(lf)
    a = ch & 0xF; b = a ^ 1
    margin = ro_frequency(ch, a, dev) - ro_frequency(ch, b, dev)
    out.append((int(bit), compute_raw_response(ch, dev), abs(margin), int(ca), int(cb)))
o = np.array(out)
agree = o[:, 0] == o[:, 1]
n = len(o)
print("CRPs                     : %d (challenge sequence mismatches: %d)" % (n, bad_ch))
print("oscillation counts       : min %d, max %d per window" % (o[:, 3:5].min(), o[:, 3:5].max()))
print("agreement with the model : %d of %d = %.2f%%" % (agree.sum(), n, 100 * agree.mean()))
if (~agree).any():
    print("disagreements at |margin|: max %d (model units)" % o[~agree, 2].max())
    print("count difference there   : max %d" % np.abs(o[~agree, 3] - o[~agree, 4]).max())
thr = int(o[~agree, 2].max()) if (~agree).any() else 0
print("agreement where |margin| > %d: %.2f%% of %d CRPs"
      % (thr, 100 * agree[o[:, 2] > thr].mean(), int((o[:, 2] > thr).sum())))
json.dump({"n": int(n), "agree": int(agree.sum()), "agree_pct": float(100 * agree.mean()),
           "max_margin_at_disagreement": thr,
           "ties_equal_counts": int((o[:, 3] == o[:, 4]).sum()),
           "count_min": int(o[:, 3:5].min()), "count_max": int(o[:, 3:5].max())},
          open("ro_timing_check.json", "w"), indent=1)
