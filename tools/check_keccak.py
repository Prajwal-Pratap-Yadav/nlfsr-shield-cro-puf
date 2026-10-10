# Writes keccak_in.txt (640 random raw bits, fixed seed) when called with
# "make", and otherwise compares keccak_out.txt from tb_keccak_layer.v
# with KeccakF200Model of generation/generate_c1_c2_datasets.py.
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
sys.path.insert(0, os.path.join(HERE, "..", "generation"))
from generate_c1_c2_datasets import KeccakF200Model
bits = np.random.default_rng(2026).integers(0, 2, 640)
if len(sys.argv) > 1 and sys.argv[1] == "make":
    open("keccak_in.txt", "w").write("".join("%d\n" % b for b in bits)); sys.exit()
k = KeccakF200Model(); ref = []
for i in range(0, 640, 64):
    ref += k.absorb_and_squeeze([int(b) for b in bits[i:i + 64]])
hw = [int(l) for l in open("keccak_out.txt") if l.strip()]
bad = sum(a != b for a, b in zip(hw, ref)) + abs(len(hw) - len(ref))
print("Keccak-f[200] layer: %d output bits, reference %d, mismatches %d" % (len(hw), len(ref), bad))
