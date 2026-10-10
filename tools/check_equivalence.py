# ============================================================
# check_equivalence.py
#
# Compares the RTL simulation dumps with the Python model.
#   rtl_out.txt   (tb_equivalence_cropuf.v): "<challenge hex> <bit>"
#   nlfsr_out.txt (tb_equivalence_nlfsr.v) : 3,000 output bits
#
#   python tools/check_equivalence.py
# ============================================================
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from puf_core import build_device, compute_raw_response, NLFSRModel, lfsr_next, LFSR_SEED

if os.path.exists("rtl_out.txt"):
    dev = build_device(0)
    lines = [l.split() for l in open("rtl_out.txt") if l.strip()]
    lf, bad_ch, bad_r = LFSR_SEED, 0, 0
    for ch_hex, bit in lines:
        ch = int(ch_hex, 16)
        bad_ch += (ch != lf)
        bad_r += (int(bit) != compute_raw_response(ch, dev))
        lf = lfsr_next(lf)
    n = len(lines)
    print("CRO-PUF : %d CRPs, challenge mismatches %d, response mismatches %d, match %.4f%%"
          % (n, bad_ch, bad_r, 100.0 * (n - bad_r) / n))

if os.path.exists("nlfsr_out.txt"):
    rtl = open("nlfsr_out.txt").read().strip()
    m = NLFSRModel()
    seed = 0x50C04CB1                      # same seed as tb_equivalence_nlfsr.v
    for k in range(31, -1, -1):
        m.seed_bit((seed >> k) & 1)
    py = "".join(str(m.next_bit()) for _ in range(len(rtl)))
    bad = sum(a != b for a, b in zip(rtl, py))
    print("NLFSR   : %d bits, mismatches %d, match %.4f%%"
          % (len(rtl), bad, 100.0 * (len(rtl) - bad) / len(rtl)))
