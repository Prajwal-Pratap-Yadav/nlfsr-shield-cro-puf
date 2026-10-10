# Checks the NLFSR-Shield layer of top_c1 in the timing simulation:
# the first 32 raw responses (c1_raw.txt) seed a Python NLFSR, and every
# later output (c1_out.txt) must equal raw XOR keystream.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
from puf_core import NLFSRModel
raw = [l.split() for l in open("c1_raw.txt") if l.strip()]
out = [l.split() for l in open("c1_out.txt") if l.strip()]
m = NLFSRModel()
for _, b in raw[:32]:
    m.seed_bit(int(b))
bad = 0
for (ch_r, r), (ch_z, z) in zip(raw[32:], out):
    bad += (ch_r != ch_z) or (int(z) != (int(r) ^ m.next_bit()))
print("C1 hardware layer: %d outputs checked, mismatches %d" % (len(out), bad))
