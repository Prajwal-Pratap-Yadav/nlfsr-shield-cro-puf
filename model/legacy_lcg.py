# ============================================================
# legacy_lcg.py  -  device generator of model versions 1 and 2
#
# Kept ONLY so that the comparison in
# experiments/independent_seed_study.py can be reproduced.
# It is not used to generate any dataset of the paper.
#
# The generator is x_{n+1} = (4259 x_n + 47) mod 2^16.
#
# It is NOT a full-period generator. For a modulus m = 2^16 the
# Hull-Dobell theorem requires (i) c odd, (ii) a - 1 divisible
# by 2, and (iii) a - 1 divisible by 4. Condition (iii) fails:
# a - 1 = 4258 = 2 x 2129. The 65,536 states split into two
# cycles of 32,768 states each (checked by lcg_cycles() below).
#
# Only the low-order bits are used (value mod 4096 for the base
# offsets, mod 2048 for the stage delays), and the low k bits of
# a power-of-two-modulus LCG form an LCG of their own, mod 2^k.
# For k = 11 this gives two cycles of only 1,024 values, so even
# devices with unrelated seeds reuse the same stage-delay values
# in shifted order once more than about 2,000 delays are drawn.
# ============================================================

import numpy as np

LCG_MULT    = 4259
LCG_INC     = 47
DEVICE_SEED = 0xBEEF       # device 0 of versions 1 and 2

NUM_ROS, STAGES = 16, 3
BASE_SPREAD, STAGE_SPREAD = 4096, 2048


def lcg_step(x):
    return (x * LCG_MULT + LCG_INC) & 0xFFFF


def build_device_lcg(seed):
    """Device parameters exactly as in puf_core.py version 2.0."""
    s = seed & 0xFFFF
    base = np.empty(NUM_ROS, np.int64)
    delta = np.empty((NUM_ROS, STAGES, 2), np.int64)
    for i in range(NUM_ROS):
        s = lcg_step(s)
        base[i] = (s % BASE_SPREAD) - BASE_SPREAD // 2
    for i in range(NUM_ROS):
        for st in range(STAGES):
            for c in range(2):
                s = lcg_step(s)
                delta[i, st, c] = (s % STAGE_SPREAD) - STAGE_SPREAD // 2
    return base, delta


def chained_seeds(n_devices):
    """Seeds of versions 1 and 2: seed(d+1) = LCG(seed(d))."""
    seeds, s = [DEVICE_SEED], DEVICE_SEED
    for _ in range(n_devices - 1):
        s = lcg_step(s)
        seeds.append(s)
    return seeds


def lcg_cycles():
    """Return the list of cycle lengths of the LCG over all 2^16 states."""
    seen = np.zeros(65536, bool)
    lengths = []
    for start in range(65536):
        if seen[start]:
            continue
        x, n = start, 0
        while not seen[x]:
            seen[x] = True
            x = lcg_step(x)
            n += 1
        lengths.append(n)
    return lengths


def lcg_position_map():
    """(cycle id, position in cycle) of every 16-bit state."""
    pos, cid = {}, 0
    for start in range(65536):
        if start in pos:
            continue
        x, i = start, 0
        while x not in pos:
            pos[x] = (cid, i)
            x = lcg_step(x)
            i += 1
        cid += 1
    return pos


if __name__ == "__main__":
    L = lcg_cycles()
    print("a - 1 = %d, (a - 1) mod 4 = %d" % (LCG_MULT - 1, (LCG_MULT - 1) % 4))
    print("cycle lengths:", L)
    s = chained_seeds(3)
    d0, d1 = build_device_lcg(s[0]), build_device_lcg(s[1])
    print("chained: base of device 1 == base of device 0 shifted by one:",
          bool(np.array_equal(d1[0][:-1], d0[0][1:])))
