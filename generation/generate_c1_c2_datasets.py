# ============================================================
# Script      : generate_c1_c2_datasets.py   [version 3.0]
# Project     : NLFSR-Shield CRO-PUF
# File        : python/metrics/generate_c1_c2_datasets.py
#
# Purpose     : Generates CRP datasets for C1 (NLFSR-Shield)
#               and C2 (Keccak-f[200]) configurations.
#
# CHANGE IN VERSION 3.0 (second revision):
#   Device d is drawn from its own independent PCG64 stream
#   (puf_core v3.0); devices are no longer chained LCG seeds.
#
# CHANGES in version 2.0 (first revision):
#   1. The PUF core and the NLFSR are no longer defined locally.
#      They are imported from puf_core.py so that a single
#      definition is shared by every script in the project.
#   2. build_device() replaces generate_fingerprints(); the
#      device object is passed to compute_raw_response() in
#      place of the flat fingerprint list.
#   3. Each dataset is written twice: a *_sequential.csv in
#      generation order for autocorrelation and NIST SP 800-22,
#      and the shuffled file for the ML attack pipeline.
#
# Outputs:
#   sim/crp_output/crp_c1_100k.csv   + crp_c1_100k_sequential.csv
#   sim/crp_output/crp_c1_1m.csv     + crp_c1_1m_sequential.csv
#   sim/crp_output/crp_c2_100k.csv   + crp_c2_100k_sequential.csv
#   sim/crp_output/crp_c2_1m.csv     + crp_c2_1m_sequential.csv
# ============================================================

import numpy as np
import pandas as pd
import os
import time

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "model"))
from puf_core import (lfsr_next, build_device, compute_raw_response,
                      device_ids, NLFSRModel, LFSR_SEED)

OUTPUT_DIR = "sim/crp_output"


# --------------------------------------------------------
# C1: NLFSR-SHIELD
# --------------------------------------------------------
def generate_c1_device(device_id, n_crps):
    """
    Generate C1 CRPs for one device.
    Phase 1: the first 32 raw PUF bits seed the NLFSR.
    Phase 2: subsequent raw bits are XORed with the NLFSR output.
    """
    dev   = build_device(device_id)
    nlfsr = NLFSRModel()
    lfsr  = LFSR_SEED

    data = np.zeros((n_crps, 17), dtype=np.uint8)

    # Phase 1: seed loading (32 bits consumed, not output)
    for _ in range(32):
        ch_int = lfsr
        lfsr   = lfsr_next(lfsr)
        nlfsr.seed_bit(compute_raw_response(ch_int, dev))

    # Phase 2: CRP generation with obfuscation
    for idx in range(n_crps):
        ch_int  = lfsr
        lfsr    = lfsr_next(lfsr)
        raw_bit = compute_raw_response(ch_int, dev)
        obf_bit = nlfsr.obfuscate(raw_bit)

        for b in range(16):
            data[idx, b] = (ch_int >> b) & 1
        data[idx, 16] = obf_bit

    return data


# --------------------------------------------------------
# C2: KECCAK-f[200]
# --------------------------------------------------------
class KeccakF200Model:
    """Python model of the Keccak-f[200] permutation."""

    RC = [0x01, 0x82, 0x8A, 0x00, 0x8B, 0x01, 0x81, 0x09,
          0x8A, 0x88, 0x09, 0x0A, 0x8B, 0x8B, 0x89, 0x03,
          0x02, 0x80]

    RHO = [0, 1, 6, 4, 3, 4, 4, 6, 7, 4, 3, 2, 3, 1, 7,
           1, 5, 7, 5, 0, 2, 2, 5, 0, 6]

    PI_MAP = [0, 15, 20, 10, 5, 1, 16, 21, 11, 6, 2, 17, 22, 12, 7,
              3, 18, 23, 13, 8, 4, 19, 24, 14, 9]

    def __init__(self):
        self.state = [0] * 25

    def _rotl8(self, val, n):
        n %= 8
        if n == 0:
            return val & 0xFF
        return ((val << n) | (val >> (8 - n))) & 0xFF

    def _permute(self):
        s = list(self.state)
        for rnd in range(18):
            C = [s[x] ^ s[5+x] ^ s[10+x] ^ s[15+x] ^ s[20+x] for x in range(5)]
            D = [C[(x+4) % 5] ^ self._rotl8(C[(x+1) % 5], 1) for x in range(5)]
            s = [s[i] ^ D[i % 5] for i in range(25)]

            tmp = [0] * 25
            for i in range(25):
                tmp[self.PI_MAP[i]] = self._rotl8(s[i], self.RHO[i])

            s2 = [0] * 25
            for y in range(5):
                for x in range(5):
                    s2[5*y+x] = tmp[5*y+x] ^ (
                        (~tmp[5*y+(x+1) % 5] & 0xFF) & tmp[5*y+(x+2) % 5])

            s2[0] ^= self.RC[rnd]
            s = s2
        self.state = s

    def absorb_and_squeeze(self, input_bits_64):
        for lane in range(8):
            byte_val = 0
            for bit in range(8):
                byte_val |= (input_bits_64[lane*8+bit] << bit)
            self.state[lane] ^= byte_val
        self._permute()
        out_bits = []
        for lane in range(8):
            for bit in range(8):
                out_bits.append((self.state[lane] >> bit) & 1)
        return out_bits


def generate_c2_device(device_id, n_crps):
    """Accumulate 64 raw bits, permute, emit 64 obfuscated bits."""
    dev    = build_device(device_id)
    keccak = KeccakF200Model()
    lfsr   = LFSR_SEED

    data = np.zeros((n_crps, 17), dtype=np.uint8)
    idx  = 0
    acc_bits, acc_chs = [], []

    while idx < n_crps:
        ch_int = lfsr
        lfsr   = lfsr_next(lfsr)
        acc_bits.append(compute_raw_response(ch_int, dev))
        acc_chs.append(ch_int)

        if len(acc_bits) == 64:
            obf_bits = keccak.absorb_and_squeeze(acc_bits)
            for ch_int_i, obf_bit in zip(acc_chs, obf_bits):
                if idx >= n_crps:
                    break
                for b in range(16):
                    data[idx, b] = (ch_int_i >> b) & 1
                data[idx, 16] = obf_bit
                idx += 1
            acc_bits, acc_chs = [], []

    return data


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("=" * 60)
    print("  C1 and C2 Dataset Generator  [v3.0]")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    col_names = [f'c{i}' for i in range(16)] + ['response']

    configs = [
        ('C1', generate_c1_device, 'crp_c1_100k.csv',   100_000,  1),
        ('C1', generate_c1_device, 'crp_c1_1m.csv',   1_000_000, 10),
        ('C2', generate_c2_device, 'crp_c2_100k.csv',   100_000,  1),
        ('C2', generate_c2_device, 'crp_c2_1m.csv',   1_000_000, 10),
    ]

    for cfg_name, gen_fn, filename, n_crps, n_devices in configs:
        out_path     = os.path.join(OUTPUT_DIR, filename)
        seq_path     = out_path.replace('.csv', '_sequential.csv')
        crps_per_dev = n_crps // n_devices
        ids          = device_ids(n_devices)

        print(f"Generating {filename}  "
              f"({n_crps:,} CRPs, {n_devices} device(s))...")
        t0 = time.time()

        combined = np.vstack([gen_fn(d, crps_per_dev) for d in ids])

        # (a) sequential copy - generation order preserved
        pd.DataFrame(combined, columns=col_names).to_csv(seq_path, index=False)

        # (b) shuffled copy - for the ML attack pipeline
        shuffled = combined.copy()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(shuffled)
        pd.DataFrame(shuffled, columns=col_names).to_csv(out_path, index=False)

        uniformity = 100.0 * combined[:, 16].sum() / len(combined)
        print(f"  Uniformity : {uniformity:.2f}%")
        print(f"  Sequential : {os.path.basename(seq_path)}")
        print(f"  Shuffled   : {os.path.basename(out_path)}")
        print(f"  Time       : {time.time() - t0:.1f}s")
        print()

    print("=" * 60)
    print("  ALL C1/C2 DATASETS GENERATED")
    print("=" * 60)


if __name__ == "__main__":
    main()
