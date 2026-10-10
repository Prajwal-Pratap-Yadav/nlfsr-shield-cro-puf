# ============================================================
# puf_core.py  -  shared behavioural model for NLFSR-Shield
# Version 3.0
#
# Every generation and attack script imports this file, so the
# device model, the challenge generator and the NLFSR are
# defined in exactly one place.
#
# ------------------------------------------------------------
# CHANGE IN VERSION 3.0 (second revision): independent devices
# ------------------------------------------------------------
# Versions 1 and 2 drew the parameters of every virtual device
# from a 16-bit Linear Congruential Generator,
#       x_{n+1} = (4259 x_n + 47) mod 2^16,
# and obtained the seed of device d+1 by applying the same LCG
# step to the seed of device d. Two problems with this were
# found while answering the second-round review:
#
#   (a) Chaining. Device d+1's seed is the first value drawn
#       for device d, so the parameter list of device d+1 is the
#       parameter list of device d shifted by one position. The
#       devices are therefore strongly related to each other;
#       neighbouring devices have an average response
#       correlation of -0.30 over the full challenge space.
#
#   (b) The LCG itself. The parameters do NOT satisfy the
#       Hull-Dobell theorem, contrary to what earlier versions
#       stated: for m = 2^16 the theorem requires 4 | (a - 1),
#       and a - 1 = 4258 is not divisible by 4. The generator
#       splits the 16-bit space into two cycles of 32,768, and,
#       like every power-of-two-modulus LCG, its low-order bits
#       have short periods. Even with non-chained seeds, the
#       devices remain measurably related.
#
# Version 3.0 therefore draws every device from its own,
# statistically independent pseudo-random stream:
#
#   stream(d) = PCG64( SeedSequence(DEVICE_ROOT_SEED,
#                                   spawn_key=(d,)) )
#
# SeedSequence derives the state of each child stream by
# hashing the root entropy together with the device index, so
# the streams of different devices do not overlap and are
# independent for all practical purposes. Device d is the same
# no matter how many devices are generated. The process
# variation is drawn from a zero-mean Gaussian distribution, the
# usual first-order model of within-die random variation, with
# the same standard deviation as the uniform windows used
# before, so the scale of the model is unchanged:
#
#   base[ro]               ~ N(0, (4096 / sqrt 12)^2)
#   delta[ro][stage][cfg]  ~ N(0, (2048 / sqrt 12)^2)
#
# all rounded to the nearest integer. DEVICE_ROOT_SEED was fixed
# once before any experiment was run and has not been tuned.
#
# The RTL simulation model (rtl/cropuf_sim_model.v, v3.0) no
# longer contains a generator. It loads the same integers from
# two hex files written by tools/export_device_hex.py, so the
# RTL and this file agree bit for bit (see README).
#
# Unchanged from version 2.0:
#   * three Configurable Inversion Unit stages per oscillator,
#     after Kareem and Dunaev [6];
#   * pair selection a = C[3:0], b = a XOR 1;
#   * the corrected NLFSR with MNL taps 30, 15, 7.
# ============================================================

import numpy as np

NUM_ROS      = 16
STAGES       = 3          # configurable inversion units per RO, per [6]
LFSR_SEED    = 0xACE1     # challenge generator seed

DEVICE_ROOT_SEED = 0x5EED20260BADF00E   # root entropy of all device streams
DEVICE_0         = 0                    # index of the reference device

BASE_SPREAD  = 4096       # width of the v2 uniform window (base)
STAGE_SPREAD = 2048       # width of the v2 uniform window (one stage)
BASE_SIGMA   = BASE_SPREAD / np.sqrt(12.0)    # = 1182.4
STAGE_SIGMA  = STAGE_SPREAD / np.sqrt(12.0)   # =  591.2

NLFSR_WIDTH  = 32
MASK32       = (1 << NLFSR_WIDTH) - 1


# ---------------- challenge generator ----------------
def lfsr_next(reg):
    """16-bit LFSR, x^16 + x^14 + x^13 + x^11 + 1."""
    fb = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) | fb) & 0xFFFF


def challenge_sequence(n=65535, seed=LFSR_SEED):
    """First n challenges of the LFSR, as a numpy int array."""
    out = np.empty(n, dtype=np.int64)
    r = seed
    for i in range(n):
        out[i] = r
        r = lfsr_next(r)
    return out


# ---------------- device model ----------------
def device_ids(n_devices):
    """Indices of the first n virtual devices (0, 1, ..., n-1)."""
    return list(range(n_devices))


def build_device(device_id):
    """Return (base, delta) for virtual device `device_id`.

    base[i]          integer frequency offset of ring oscillator i
    delta[i][s][b]   integer delay contributed by CIU stage s of
                     oscillator i when that stage is configured to b

    Each device has its own independent PCG64 stream, derived from
    DEVICE_ROOT_SEED and the device index by numpy's SeedSequence.
    """
    if not (isinstance(device_id, (int, np.integer)) and device_id >= 0):
        raise ValueError("device_id must be a non-negative integer index")
    ss = np.random.SeedSequence(DEVICE_ROOT_SEED, spawn_key=(int(device_id),))
    g = np.random.default_rng(ss)
    base = np.rint(g.normal(0.0, BASE_SIGMA, NUM_ROS)).astype(np.int64)
    delta = np.rint(g.normal(0.0, STAGE_SIGMA, (NUM_ROS, STAGES, 2))).astype(np.int64)
    return base, delta


def cfg_bit(ch_int, ro, stage):
    """Challenge bit driving `stage` of oscillator `ro`.

    The stride of 5 is coprime with 16, so the three stages of one
    oscillator read three different challenge bits and, across the
    sixteen oscillators, every challenge bit drives some stage.
    """
    return (ch_int >> ((ro + 5 * stage) & 0xF)) & 1


def ro_frequency(ch_int, ro, dev):
    base, delta = dev
    f = int(base[ro])
    for st in range(STAGES):
        f += int(delta[ro][st][cfg_bit(ch_int, ro, st)])
    return f


def compute_raw_response(ch_int, dev):
    """One raw PUF response bit."""
    sel_a = ch_int & 0xF
    sel_b = sel_a ^ 0x1
    return 1 if ro_frequency(ch_int, sel_a, dev) > \
                ro_frequency(ch_int, sel_b, dev) else 0


def response_vector(challenges, dev):
    """Vectorised compute_raw_response over an array of challenges.
    Gives exactly the same bits as the scalar function."""
    ch = np.asarray(challenges, dtype=np.int64)
    base, delta = dev
    freq = np.repeat(np.asarray(base, np.int64)[None, :], len(ch), 0)
    for ro in range(NUM_ROS):
        for s in range(STAGES):
            b = (ch >> ((ro + 5 * s) & 15)) & 1
            freq[:, ro] += np.where(b == 1, delta[ro, s, 1], delta[ro, s, 0])
    a = ch & 0xF
    rows = np.arange(len(ch))
    return (freq[rows, a] > freq[rows, a ^ 1]).astype(np.uint8)


# ---------------- export for the RTL simulation model ----------------
def write_device_hex(dev, base_path, delta_path):
    """Write the device parameters as 16-bit two's-complement hex words,
    one per line, in the order read by $readmemh in cropuf_sim_model.v:
      base file : base[0] ... base[15]
      delta file: delta[ro][stage][cfg] at index ro*6 + stage*2 + cfg
    """
    base, delta = dev
    with open(base_path, "w") as f:
        for v in base:
            f.write("%04X\n" % (int(v) & 0xFFFF))
    with open(delta_path, "w") as f:
        for ro in range(NUM_ROS):
            for s in range(STAGES):
                for c in range(2):
                    f.write("%04X\n" % (int(delta[ro, s, c]) & 0xFFFF))


# ---------------- corrected NLFSR (unchanged from v2.0) ----------------
class NLFSRModel:
    """32-bit NLFSR with a maximally non-linear three-variable combining
    function. Mirrors nlfsr_core.v."""

    def __init__(self, width=NLFSR_WIDTH):
        self.width = width
        self.state = 0
        self.ready = False
        self.seed_count = 0

    def reset(self):
        self.state = 0
        self.ready = False
        self.seed_count = 0

    def _linear_fb(self):
        s = self.state
        # x^32 + x^22 + x^2 + x + 1 -> taps 31, 21, 1, 0
        return ((s >> 31) ^ (s >> 21) ^ (s >> 1) ^ s) & 1

    def _mnl_fb(self):
        # MNL taps 30, 15, 7. Bit 31 must appear only in the linear
        # term; if it also appears in a product term the state map is
        # not a bijection and the period collapses (1,832 with 31, 15, 7).
        a = (self.state >> 30) & 1
        b = (self.state >> 15) & 1
        c = (self.state >> 7) & 1
        return (a & b) ^ (b & c) ^ (a & c)

    def _combined_fb(self):
        return self._linear_fb() ^ self._mnl_fb()

    def seed_bit(self, bit):
        if self.ready:
            return True
        self.state = ((self.state << 1) & MASK32) | (bit & 1)
        self.seed_count += 1
        if self.seed_count >= self.width:
            self.ready = True
        return self.ready

    def next_bit(self):
        out = self._combined_fb()
        self.state = ((self.state << 1) & MASK32) | out
        return out

    def obfuscate(self, raw_bit):
        return raw_bit ^ self.next_bit()


# ---------------- self-test ----------------
if __name__ == "__main__":
    chs = challenge_sequence()
    devs = [build_device(d) for d in device_ids(30)]
    bits = np.array([response_vector(chs, d) for d in devs])

    # scalar and vectorised functions must agree
    d0 = devs[0]
    scalar = np.array([compute_raw_response(int(c), d0) for c in chs[:5000]])
    assert np.array_equal(scalar, bits[0][:5000]), "vectorised model mismatch"

    # device d must not depend on how many devices are generated
    assert all(np.array_equal(build_device(7)[k], devs[7][k]) for k in (0, 1))

    uniq = [np.mean(bits[i] != bits[j])
            for i in range(len(devs)) for j in range(i + 1, len(devs))]
    C = np.corrcoef(bits.astype(float))
    neigh = np.mean([C[i, i + 1] for i in range(29)])

    diff = [np.mean([compute_raw_response(int(c) ^ (1 << p), d0) !=
                     compute_raw_response(int(c), d0) for p in range(16)])
            for c in chs[:5000]]

    # Single-bit sensitivity over the full challenge space.
    # A challenge bit is "effective" on a device if flipping it changes the
    # response for at least one challenge. On an individual device a CIU bit
    # can be ineffective when the base-frequency gap of the oscillator pair
    # it configures exceeds the total configurable delay (a stable pair),
    # so the model-level requirement is that every bit is effective on some
    # device; the per-device average is reported.
    eff_dev = np.array([[np.any(response_vector(chs ^ (1 << p), d) !=
                                response_vector(chs, d)) for p in range(16)]
                        for d in devs])
    per_dev = eff_dev.sum(1)
    union = int(eff_dev.any(0).sum())

    print("Device streams          : PCG64 / SeedSequence, root 0x%X" % DEVICE_ROOT_SEED)
    print("Uniformity (30 devices) : %.2f%%" % (100 * bits.mean()))
    print("Uniformity (device 0)   : %.2f%%" % (100 * bits[0].mean()))
    print("Uniqueness              : %.2f%%" % (100 * np.mean(uniq)))
    print("Neighbour correlation   : %+.4f" % neigh)
    print("Diffuseness (device 0)  : %.2f%%" % (100 * np.mean(diff)))
    print("Effective challenge bits: %d of 16 across devices; "
          "per device mean %.2f (min %d, max %d)"
          % (union, per_dev.mean(), per_dev.min(), per_dev.max()))
    assert union == 16, "CIU is inert - every challenge bit must be effective"

    n = NLFSRModel()
    coll = 0
    for k in range(100000):
        s = (k * 2654435761) & MASK32
        n.state = s;              a = n._combined_fb()
        n.state = s ^ 0x80000000; b = n._combined_fb()
        if (((s << 1) & MASK32) | a) == ((((s ^ 0x80000000) << 1) & MASK32) | b):
            coll += 1
    print("NLFSR bit-31 collisions : %d of 100000 (must be 0)" % coll)
    assert coll == 0, "NLFSR state map is not a bijection"
    print("\nAll checks passed.")
