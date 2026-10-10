# ============================================================
# Script      : ablation_study.py   [CORRECTED VERSION]
# Project     : NLFSR-Shield CRO-PUF
# File        : python/ml_attacks/ablation_study.py
#
# Purpose     : Ablation study - isolates the contribution of
#               each component of the NLFSR-Shield.
#
# CHANGES vs the previous version:
#   1. The PUF device model is imported from puf_core.py
#      instead of being redefined locally, so this script uses
#      the same three-stage CIU as the dataset generators.
#   2. Variants A3 and A4 use the corrected MNL taps (30,15,7).
#      With the previous taps (31,15,7) bit 31 appeared inside
#      a product term, the state map was not a bijection, and
#      the period collapsed to 1,832 for every device seed.
#   3. "Bent" is replaced by "MNL" (maximally non-linear)
#      throughout, since a three-variable function cannot be
#      bent.
#
# Five variants, DNN/Adam at 100K and 1M CRPs:
#   A0 : Bare CRO-PUF (C0 baseline) - control
#   A1 : CRO-PUF + 16-bit linear LFSR
#   A2 : CRO-PUF + 32-bit register, linear feedback only
#   A3 : CRO-PUF + 32-bit register + MNL function (= C1)
#   A4 : Register alone driven by random input, not the PUF
# ============================================================

import numpy as np
import pandas as pd
import os, sys, time, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'metrics'))
from puf_core import (lfsr_next, build_device, compute_raw_response,
                      LFSR_SEED, DEVICE_0, MASK32)

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

OUTPUT_DIR = "results/ml"
SEEDS      = [42, 123, 456]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------------
# FEEDBACK HELPERS
# --------------------------------------------------------
def linear_fb32(state):
    """x^32 + x^22 + x^2 + x + 1 -> taps 31, 21, 1, 0."""
    return ((state >> 31) ^ (state >> 21) ^ (state >> 1) ^ state) & 1


def mnl_fb32(state):
    """Maximally non-linear three-variable combining function.
    CORRECTED taps 30, 15, 7: bit 31 must appear only in the
    linear term or the state map stops being a bijection."""
    a = (state >> 30) & 1
    b = (state >> 15) & 1
    c = (state >> 7) & 1
    return (a & b) ^ (b & c) ^ (a & c)


def pack(data, i, ch, resp):
    for b in range(16):
        data[i, b] = (ch >> b) & 1
    data[i, 16] = resp


# --------------------------------------------------------
# DNN ATTACK
# --------------------------------------------------------
def build_dnn(seed):
    tf.random.set_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(16,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def run_attack(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=seed, stratify=y_tr)
    sc = StandardScaler()
    model = build_dnn(seed)
    model.fit(sc.fit_transform(X_tr), y_tr,
              validation_data=(sc.transform(X_val), y_val),
              epochs=100, batch_size=256,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=10, restore_best_weights=True)],
              verbose=0)
    acc = model.evaluate(sc.transform(X_te), y_te, verbose=0)[1]
    keras.backend.clear_session()
    return acc * 100.0


# --------------------------------------------------------
# VARIANT GENERATORS
# --------------------------------------------------------
def gen_a0(dev, n_crps):
    """A0: bare CRO-PUF, no obfuscation."""
    data = np.zeros((n_crps, 17), dtype=np.uint8)
    lfsr = LFSR_SEED
    for i in range(n_crps):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        pack(data, i, ch, compute_raw_response(ch, dev))
    return data


def gen_a1(dev, n_crps):
    """A1: CRO-PUF + 16-bit linear LFSR obfuscation."""
    data = np.zeros((n_crps, 17), dtype=np.uint8)
    lfsr_ch, lfsr_obf = LFSR_SEED, 0xDEAD
    for _ in range(16):
        ch = lfsr_ch; lfsr_ch = lfsr_next(lfsr_ch)
        lfsr_obf = ((lfsr_obf << 1) | compute_raw_response(ch, dev)) & 0xFFFF
    for i in range(n_crps):
        ch = lfsr_ch; lfsr_ch = lfsr_next(lfsr_ch)
        r = compute_raw_response(ch, dev)
        obf_bit = (lfsr_obf >> 15) & 1
        fb = ((lfsr_obf >> 15) ^ (lfsr_obf >> 13) ^
              (lfsr_obf >> 12) ^ (lfsr_obf >> 10)) & 1
        lfsr_obf = ((lfsr_obf << 1) | fb) & 0xFFFF
        pack(data, i, ch, r ^ obf_bit)
    return data


def gen_a2(dev, n_crps):
    """A2: CRO-PUF + 32-bit register, linear feedback only."""
    data = np.zeros((n_crps, 17), dtype=np.uint8)
    lfsr, state = LFSR_SEED, 0
    for _ in range(32):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        state = ((state << 1) | compute_raw_response(ch, dev)) & MASK32
    for i in range(n_crps):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        r = compute_raw_response(ch, dev)
        out = linear_fb32(state)
        state = ((state << 1) | out) & MASK32
        pack(data, i, ch, r ^ out)
    return data


def gen_a3(dev, n_crps):
    """A3: full NLFSR-Shield, linear + MNL feedback (= C1)."""
    data = np.zeros((n_crps, 17), dtype=np.uint8)
    lfsr, state = LFSR_SEED, 0
    for _ in range(32):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        state = ((state << 1) | compute_raw_response(ch, dev)) & MASK32
    for i in range(n_crps):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        r = compute_raw_response(ch, dev)
        out = linear_fb32(state) ^ mnl_fb32(state)
        state = ((state << 1) | out) & MASK32
        pack(data, i, ch, r ^ out)
    return data


def gen_a4(dev, n_crps):
    """A4: register alone, driven by random input rather than the PUF."""
    data = np.zeros((n_crps, 17), dtype=np.uint8)
    lfsr, state = LFSR_SEED, 0xABCDEF01
    rng = np.random.default_rng(seed=99)
    for i in range(n_crps):
        ch = lfsr; lfsr = lfsr_next(lfsr)
        r_rand = int(rng.integers(0, 2))
        out = linear_fb32(state) ^ mnl_fb32(state)
        state = ((state << 1) | out) & MASK32
        pack(data, i, ch, r_rand ^ out)
    return data


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("=" * 60)
    print("  NLFSR-Shield Ablation Study  [corrected model]")
    print("=" * 60)

    dev = build_device(DEVICE_0)   # reference device (v3.0)

    variants = [
        ('A0', 'Bare CRO-PUF (baseline)',        gen_a0),
        ('A1', 'CRO-PUF + 16-bit linear LFSR',   gen_a1),
        ('A2', 'CRO-PUF + 32-bit linear only',   gen_a2),
        ('A3', 'CRO-PUF + 32-bit + MNL (C1)',    gen_a3),
        ('A4', 'Register alone (random input)',  gen_a4),
    ]

    all_results = []

    for n_crps in [100_000, 1_000_000]:
        print(f"\n{'-'*60}")
        print(f"  Dataset size: {n_crps:,} CRPs")
        print(f"{'-'*60}")

        for var_id, var_name, gen_fn in variants:
            print(f"\n  {var_id}: {var_name}")
            t0 = time.time()
            data = gen_fn(dev, n_crps)
            X = data[:, :16].astype(np.float32)
            y = data[:, 16].astype(np.int32)
            print(f"  Generated: bal={100*y.mean():.1f}%  "
                  f"({time.time()-t0:.1f}s)")

            accs = []
            for seed in SEEDS:
                t0 = time.time()
                acc = run_attack(X, y, seed)
                accs.append(acc)
                print(f"    seed={seed}: {acc:.2f}%  ({time.time()-t0:.1f}s)")
                all_results.append(dict(variant=var_id, name=var_name,
                                        n_crps=n_crps, seed=seed,
                                        test_acc=round(acc, 4)))
            print(f"  Mean: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")

    df = pd.DataFrame(all_results)
    summary = df.groupby(['variant', 'name', 'n_crps'])['test_acc'] \
                .agg(['mean', 'std']).reset_index()

    print("\n" + "=" * 60)
    print("  ABLATION RESULTS - PAPER TABLE V")
    print("=" * 60)
    print(f"  {'Variant':<6} {'Description':<32} {'100K':>8} {'1M':>8}")
    print("  " + "-" * 56)
    names = {v[0]: v[1] for v in variants}
    for var_id in ['A0', 'A1', 'A2', 'A3', 'A4']:
        r100 = summary[(summary['variant'] == var_id) &
                       (summary['n_crps'] == 100_000)]
        r1m = summary[(summary['variant'] == var_id) &
                      (summary['n_crps'] == 1_000_000)]
        a100 = f"{r100['mean'].values[0]:.2f}%" if len(r100) else 'N/A'
        a1m = f"{r1m['mean'].values[0]:.2f}%" if len(r1m) else 'N/A'
        print(f"  {var_id:<6} {names[var_id]:<32} {a100:>8} {a1m:>8}")
    print("=" * 60)

    df.to_csv(os.path.join(OUTPUT_DIR, 'ablation_results.csv'), index=False)
    summary.to_csv(os.path.join(OUTPUT_DIR, 'ablation_summary.csv'), index=False)
    print(f"\n  Saved: {OUTPUT_DIR}/ablation_results.csv")
    print(f"  Saved: {OUTPUT_DIR}/ablation_summary.csv")


if __name__ == "__main__":
    main()
