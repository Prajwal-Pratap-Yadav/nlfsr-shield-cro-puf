# ============================================================
# design_aware_analysis.py
#
# Measures the properties of the CRO-PUF and of the NLFSR-Shield
# keystream that bound what an attacker who KNOWS the construction
# can do, as opposed to the data-driven ML attacker of Section 5.
# This supports the honest limitation of the security claims asked
# for by Reviewer 4 (second round, comment 2).
#
# It measures four public, structural facts. None of them is a new
# cryptanalytic technique; they are standard properties of a short
# shift-register keystream and of this particular CRO-PUF, written
# out so the paper can state them exactly.
#
#   (1) Effective challenge classes of the CRO-PUF.
#       A response compares oscillators a = C[3:0] and b = a^1,
#       each driven by three challenge bits, so the 16-bit
#       challenge acts through far fewer than 2^16 distinct inputs.
#
#   (2) The NLFSR keystream obeys a fixed public recurrence.
#       k_t = s31^s21^s1^s0 ^ maj(s30,s15,s7) and the state is a
#       shift register, so k_t is a fixed Boolean function of the
#       previous 32 keystream bits. An observer who reads the
#       C1 output stream IN ORDER therefore has a self-contained
#       sequence, unlike the row-shuffled datasets used for the
#       ML evaluation in Section 5.
#
#   (3) Entropy of the 32-bit seed.
#       The seed is 32 raw PUF bits, but those bits come from only
#       a few oscillator comparisons, so the seed carries far less
#       than 32 bits of entropy across devices.
#
#   (4) Period of the NLFSR for the adopted device seeds.
#
#     python experiments/design_aware_analysis.py
# Output: experiments/design_aware_results.json
# ============================================================
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import puf_core as P

res = {}
CH = P.challenge_sequence()

# ---------- (1) effective challenge classes ----------
def class_bits(a):
    b = a ^ 1
    return sorted({(a + 5 * s) & 15 for s in range(3)} | {(b + 5 * s) & 15 for s in range(3)})

classes = 0
for a in range(16):
    free = [p for p in class_bits(a) if p not in (0, 1, 2, 3)]   # bits 0..3 are the select value a
    classes += 2 ** len(free)

def eff_key(c):
    a = c & 0xF
    k = a
    for p in class_bits(a):
        k = (k << 1) | ((c >> p) & 1)
    return k

keys = np.array([eff_key(int(c)) for c in CH])
# a device's response must be constant within each class (sanity check)
incons = 0
for d in range(5):
    r = P.response_vector(CH, P.build_device(d))
    for k in np.unique(keys):
        v = r[keys == k]
        incons += int(v.min() != v.max())
res["effective_classes"] = int(classes)
res["classes_seen_in_lfsr_sequence"] = int(len(np.unique(keys)))
res["class_response_inconsistencies"] = int(incons)
print("(1) CRO-PUF effective challenge classes: %d (of 2^16); seen in the LFSR run: %d; inconsistencies: %d"
      % (classes, len(np.unique(keys)), incons))

# ---------- (2) the keystream is a fixed function of its last 32 bits ----------
def keystream(seed_state, n):
    s = seed_state & 0xFFFFFFFF
    out = np.empty(n, np.uint8)
    for t in range(n):
        lin = (s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s
        a, b, c = (s >> 30) & 1, (s >> 15) & 1, (s >> 7) & 1
        k = (lin ^ ((a & b) ^ (b & c) ^ (a & c))) & 1
        out[t] = k
        s = ((s << 1) | k) & 0xFFFFFFFF
    return out

rng = np.random.default_rng(12345)
ks = keystream(int(rng.integers(0, 1 << 32)), 100000)
# predict k_t from the 32 preceding output bits using the public recurrence:
# since s_t = (k_{t-1}..k_{t-32}) bit-reversed, re-derive k_t and compare
ok = 0
tot = 0
s = 0
for t in range(len(ks)):
    if t >= 32:
        lin = (s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s
        a, b, c = (s >> 30) & 1, (s >> 15) & 1, (s >> 7) & 1
        pred = (lin ^ ((a & b) ^ (b & c) ^ (a & c))) & 1
        ok += int(pred == ks[t]); tot += 1
    s = ((s << 1) | int(ks[t])) & 0xFFFFFFFF
res["keystream_recurrence_accuracy"] = 100.0 * ok / tot
print("(2) next keystream bit predicted from the previous 32 output bits: %.2f%% of %d"
      % (100.0 * ok / tot, tot))

# ---------- (3) seed entropy across devices (G4 distribution) ----------
t0 = time.time()
N = 200000
g = np.random.default_rng(np.random.SeedSequence(777))
base = np.rint(g.normal(0, P.BASE_SIGMA, (N, 16)))
delta = np.rint(g.normal(0, P.STAGE_SIGMA, (N, 16, 3, 2)))
seeds = np.zeros(N, dtype=np.uint64)
for c in CH[:32]:
    c = int(c); a = c & 15; b = a ^ 1
    fa = base[:, a].copy(); fb = base[:, b].copy()
    for s_ in range(3):
        fa += delta[:, a, s_, (c >> ((a + 5 * s_) & 15)) & 1]
        fb += delta[:, b, s_, (c >> ((b + 5 * s_) & 15)) & 1]
    seeds = (seeds << np.uint64(1)) | (fa > fb).astype(np.uint64)
u, cnt = np.unique(seeds, return_counts=True)
p = cnt / N
res["seed_distinct"] = int(len(u))
res["seed_devices"] = N
res["seed_collision_entropy_bits"] = float(-np.log2(np.sum(p ** 2)))
res["seed_min_entropy_bits"] = float(-np.log2(p.max()))
res["seed_identical_pairs"] = int((cnt * (cnt - 1) // 2).sum())
print("(3) seed entropy over %d devices: collision H2 = %.1f bits, min-entropy = %.1f bits (of 32); distinct seeds %d"
      % (N, res["seed_collision_entropy_bits"], res["seed_min_entropy_bits"], len(u)))

# ---------- (4) NLFSR seeds of the adopted devices (periods via nlfsr_period_seeds.c) ----------
dev_seeds = []
for d in range(10):
    r = P.response_vector(CH[:32], P.build_device(d))
    s = 0
    for bit in r:
        s = ((s << 1) | int(bit)) & 0xFFFFFFFF
    dev_seeds.append(s)
res["device_seeds_hex"] = ["%08X" % s for s in dev_seeds]
with open(os.path.join(HERE, "device_seeds.txt"), "w") as f:
    for s in dev_seeds:
        f.write("%08X\n" % s)
print("(4) NLFSR seeds of devices 0-9 written to device_seeds.txt; run nlfsr_period_seeds for periods")

json.dump(res, open(os.path.join(HERE, "design_aware_results.json"), "w"), indent=1)
print("done in %.0fs" % (time.time() - t0))
