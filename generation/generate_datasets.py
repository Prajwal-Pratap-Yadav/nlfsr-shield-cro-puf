# ============================================================
# Script      : generate_datasets.py   [version 3.0]
# Project     : NLFSR-Shield CRO-PUF
# File        : generation/generate_datasets.py
#
# Purpose     : Generates the four C0 (unprotected baseline)
#               dataset files required for the ML experiments.
#
# CHANGE IN VERSION 3.0 (second revision):
#   Virtual devices are no longer chained LCG seeds. Device d is
#   drawn from its own independent PCG64 stream (puf_core v3.0,
#   build_device(d)), so the devices of the 500K, 1M and 3M
#   datasets are statistically independent of each other.
#
# CHANGES in version 2.0 (first revision):
#   1. The device model is no longer defined locally. It is
#      imported from puf_core.py, so generate_datasets.py,
#      generate_c1_c2_datasets.py, ablation_study.py and
#      measured_dataset_attack.py all share one definition.
#   2. build_device() replaces generate_fingerprints(). Each
#      ring oscillator now has three configurable inversion
#      stages, as specified in reference [6], instead of a
#      single flat fingerprint with an inert +256 offset.
#      All sixteen challenge bits are now effective.
#   3. Each dataset is written twice: a *_sequential.csv in
#      generation order for the autocorrelation and NIST
#      SP 800-22 analysis, and the shuffled file used by the
#      ML attack pipeline.
#   4. The cross-check against an RTL capture is informational.
#      The authoritative RTL-to-Python check is
#      tools/check_equivalence.py (see README).
#
# Outputs     : sim/crp_output/crp_c0_100k.csv  (+ _sequential)
#               sim/crp_output/crp_c0_500k.csv  (+ _sequential)
#               sim/crp_output/crp_c0_1m.csv    (+ _sequential)
#               sim/crp_output/crp_c0_3m.csv    (+ _sequential)
#               sim/crp_output/dataset_manifest.txt
# ============================================================

import numpy as np
import pandas as pd
import hashlib
import os
import time

import sys as _sys, os as _os; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "model"))
from puf_core import (lfsr_next, build_device, compute_raw_response,
                      device_ids, LFSR_SEED, DEVICE_0, DEVICE_ROOT_SEED,
                      NUM_ROS, STAGES, BASE_SIGMA, STAGE_SIGMA)

RAW_FILE   = "sim/crp_output/crp_c0_100k_raw.csv"
OUTPUT_DIR = "sim/crp_output"

DATASET_CONFIG = {
    "crp_c0_100k.csv": {"n_crps":   100_000, "n_devices":  1},
    "crp_c0_500k.csv": {"n_crps":   500_000, "n_devices":  5},
    "crp_c0_1m.csv":   {"n_crps": 1_000_000, "n_devices": 10},
    "crp_c0_3m.csv":   {"n_crps": 3_000_000, "n_devices": 30},
}


# --------------------------------------------------------
# GENERATE CRPs FOR ONE VIRTUAL DEVICE
# --------------------------------------------------------
def generate_device_crps(device_id, n_crps):
    """Returns an array of shape (n_crps, 17): 16 challenge bits
    in columns 0-15 and the response bit in column 16."""
    dev = build_device(device_id)
    data = np.zeros((n_crps, 17), dtype=np.uint8)

    lfsr = LFSR_SEED
    for i in range(n_crps):
        challenge_int = lfsr
        lfsr = lfsr_next(lfsr)
        for b in range(16):
            data[i, b] = (challenge_int >> b) & 1
        data[i, 16] = compute_raw_response(challenge_int, dev)

    return data


# --------------------------------------------------------
# CROSS-CHECK AGAINST THE VIVADO RAW CAPTURE
# --------------------------------------------------------
def check_against_vivado(raw_file):
    """Compare device 0 against a raw CSV captured from the RTL
    simulation of cropuf_sim_model.v (v3.0, loaded with the hex
    files of device 0). Reported for information only.
    """
    print("Cross-checking device 0 against the Vivado capture...")

    if not os.path.exists(raw_file):
        print(f"  Raw file not found: {raw_file}  (skipping)")
        return

    vivado_df = pd.read_csv(raw_file)
    n_check = len(vivado_df)
    py_data = generate_device_crps(DEVICE_0, n_check)

    matches = int(np.sum(vivado_df['response'].values == py_data[:, 16]))
    pct = 100.0 * matches / n_check

    print(f"  CRPs checked  : {n_check:,}")
    print(f"  Match rate    : {pct:.4f}%")

    if pct == 100.0:
        print("  STATUS        : RTL and Python agree.")
    else:
        print("  STATUS        : DISAGREEMENT. The capture was probably")
        print("  produced by an older RTL version. Re-run the capture")
        print("  with cropuf_sim_model.v v3.0 and device_000_*.hex.")


# --------------------------------------------------------
# STATISTICS
# --------------------------------------------------------
def print_statistics(name, data):
    responses = data[:, 16]
    ones = int(np.sum(responses))
    total = len(responses)
    uniformity = 100.0 * ones / total
    print(f"  Dataset       : {name}")
    print(f"  Total CRPs    : {total:,}")
    print(f"  Ones          : {ones:,}  ({uniformity:.2f}%)")
    print(f"  Uniformity    : {uniformity:.2f}%  (ideal=50.00%)")
    print(f"  Deviation     : {abs(50.0 - uniformity):.2f}%")


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("=" * 56)
    print("  NLFSR-Shield - CRP Dataset Generator  [v3.0]")
    print("=" * 56)
    print(f"  Device model : {NUM_ROS} ROs x {STAGES} configurable stages")
    print(f"  Variation    : Gaussian, sigma base {BASE_SIGMA:.1f}, sigma stage {STAGE_SIGMA:.1f}")
    print(f"  Devices      : independent PCG64 streams, root 0x{DEVICE_ROOT_SEED:X}")
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    check_against_vivado(RAW_FILE)
    print()

    manifest_lines = []
    col_names = [f'c{i}' for i in range(16)] + ['response']

    for filename, config in DATASET_CONFIG.items():
        n_crps = config['n_crps']
        n_devices = config['n_devices']
        out_path = os.path.join(OUTPUT_DIR, filename)
        seq_path = out_path.replace('.csv', '_sequential.csv')

        print(f"Generating {filename}  "
              f"({n_crps:,} CRPs, {n_devices} device(s))...")
        t_start = time.time()

        ids = device_ids(n_devices)
        crps_per_device = n_crps // n_devices

        all_data = []
        for d_idx in ids:
            all_data.append(generate_device_crps(d_idx, crps_per_device))
            if (d_idx + 1) % 5 == 0 or d_idx == 0:
                print(f"  Device {d_idx+1:2d}/{n_devices} done "
                      f"(device id {d_idx})")

        combined = np.vstack(all_data)

        # (a) sequential copy - generation order preserved, required
        #     for autocorrelation and NIST SP 800-22
        pd.DataFrame(combined, columns=col_names).to_csv(seq_path, index=False)

        # (b) shuffled copy - used by the ML attack pipeline so that
        #     device boundaries are not visible to the models
        shuffled = combined.copy()
        rng = np.random.default_rng(seed=42)
        rng.shuffle(shuffled)
        pd.DataFrame(shuffled, columns=col_names).to_csv(out_path, index=False)

        t_elapsed = time.time() - t_start
        file_mb = os.path.getsize(out_path) / (1024 * 1024)
        sha256 = sha256_file(out_path)

        print_statistics(filename, combined)
        print(f"  Sequential    : {os.path.basename(seq_path)}")
        print(f"  File size     : {file_mb:.1f} MB")
        print(f"  Time          : {t_elapsed:.1f}s")
        print(f"  SHA-256       : {sha256[:32]}...")
        print()

        manifest_lines.append(
            f"{filename}|{n_crps}|{n_devices}|{file_mb:.1f}MB|{sha256}")

    manifest_path = os.path.join(OUTPUT_DIR, "dataset_manifest.txt")
    with open(manifest_path, 'w') as f:
        f.write("NLFSR-Shield CRP Dataset Manifest\n")
        f.write(f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LFSR Seed : 0x{LFSR_SEED:04X}\n")
        f.write(f"Devices   : independent PCG64 streams (SeedSequence)\n")
        f.write(f"Root seed : 0x{DEVICE_ROOT_SEED:X}\n")
        f.write(f"Device ids: 0 .. n-1 (device 0 = reference device)\n")
        f.write(f"ROs       : {NUM_ROS}\n")
        f.write(f"CIU stages per RO : {STAGES}\n")
        f.write(f"Sigma base        : {BASE_SIGMA:.3f}\n")
        f.write(f"Sigma stage       : {STAGE_SIGMA:.3f}\n")
        f.write("-" * 56 + "\n")
        f.write("filename|crps|devices|size|sha256\n")
        for line in manifest_lines:
            f.write(line + "\n")

    print("=" * 56)
    print("  ALL DATASETS GENERATED SUCCESSFULLY")
    print(f"  Manifest: {manifest_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
