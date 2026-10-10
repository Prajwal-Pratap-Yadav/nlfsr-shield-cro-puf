# ============================================================
# Script      : measured_dataset_attack.py   [CORRECTED VERSION]
# Project     : NLFSR-Shield CRO-PUF
# File        : attacks/measured_dataset_attack.py
#
# Purpose     : ML attack evaluation on the real FPGA CRP
#               dataset and its NLFSR-obfuscated version.
#
# Source      : Abulibdeh et al., IEEE DataPort 2023
#               DOI: 10.21227/6ssx-ft56
#
# CHANGES vs the previous version:
#   The NLFSR is imported from puf_core.py rather than
#   redefined here, so the corrected MNL taps (30,15,7) are
#   used. With the previous taps (31,15,7) the state map was
#   not a bijection and the period collapsed to 1,832.
#
# Conditions:
#   RH_C0     : raw real hardware CRPs, no obfuscation
#   RH_C1     : NLFSR-Shield applied to the response stream
#   RH_MASKED : raw real hardware, good bits only [1,5,6,9,13]
# ============================================================

import numpy as np
import pandas as pd
import os, sys, time, warnings, json
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'metrics'))
from puf_core import NLFSRModel

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

DATA_DIR   = "sim/crp_output/real_hardware"
OUTPUT_DIR = "results/ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS     = [42, 123, 456, 789, 1234]
GOOD_BITS = [1, 5, 6, 9, 13]


def apply_nlfsr_to_responses(responses):
    """First 32 bits seed the NLFSR; the remainder are obfuscated."""
    nlfsr = NLFSRModel()
    obf = np.zeros(len(responses), dtype=np.uint8)
    for i, r in enumerate(responses):
        if not nlfsr.ready:
            nlfsr.seed_bit(int(r))
            obf[i] = int(r)          # seeding phase, kept raw
        else:
            obf[i] = nlfsr.obfuscate(int(r))
    return obf


def run_lr(X_tr, X_te, y_tr, y_te, seed):
    sc = StandardScaler()
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    clf.fit(sc.fit_transform(X_tr), y_tr)
    return accuracy_score(y_te, clf.predict(sc.transform(X_te)))


def run_dnn(X_tr, X_val, X_te, y_tr, y_val, y_te, seed):
    tf.random.set_seed(seed)
    sc = StandardScaler()
    X_trs = sc.fit_transform(X_tr)
    model = keras.Sequential([
        keras.layers.Input(shape=(X_trs.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_trs, y_tr,
              validation_data=(sc.transform(X_val), y_val),
              epochs=50, batch_size=256,
              callbacks=[keras.callbacks.EarlyStopping(
                  patience=8, restore_best_weights=True)],
              verbose=0)
    acc = model.evaluate(sc.transform(X_te), y_te, verbose=0)[1]
    keras.backend.clear_session()
    return acc


def run_xgb(X_tr, X_val, X_te, y_tr, y_val, y_te, seed):
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                            learning_rate=0.05, random_state=seed,
                            eval_metric='logloss', verbosity=0)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return accuracy_score(y_te, clf.predict(X_te))


def evaluate_dataset(label, X, y):
    print(f"\n  {label}")
    print(f"  Shape: {X.shape}  Balance: {100*y.mean():.1f}%")
    all_res = {'LR': [], 'DNN_adam': [], 'XGBoost': []}

    for seed in SEEDS:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)

        t0 = time.time()
        a = run_lr(X_tr, X_te, y_tr, y_te, seed)
        all_res['LR'].append(a * 100)
        print(f"    LR        seed={seed} {a*100:.2f}%  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        a = run_dnn(X_tr, X_val, X_te, y_tr, y_val, y_te, seed)
        all_res['DNN_adam'].append(a * 100)
        print(f"    DNN/Adam  seed={seed} {a*100:.2f}%  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        a = run_xgb(X_tr, X_val, X_te, y_tr, y_val, y_te, seed)
        all_res['XGBoost'].append(a * 100)
        print(f"    XGBoost   seed={seed} {a*100:.2f}%  ({time.time()-t0:.1f}s)")

    print(f"\n  {label} Summary:")
    for m, accs in all_res.items():
        print(f"    {m:<12}: {np.mean(accs):.2f}% +/- {np.std(accs):.2f}%")

    return {m: {'mean': round(float(np.mean(a)), 2),
                'std': round(float(np.std(a)), 2)}
            for m, a in all_res.items()}


def main():
    print("=" * 60)
    print("  Real Hardware ML Attack - Full Suite  [corrected NLFSR]")
    print("  Source: Abulibdeh et al. DOI:10.21227/6ssx-ft56")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, 'crp_real_fpga_100k.csv'))
    CH_COLS = [f'c{i}' for i in range(32)]
    X = df[CH_COLS].values.astype(np.float32)
    y = df['response'].values.astype(np.int32)

    all_results = {}

    print("\n" + "-" * 60)
    print("  RH_C0: Raw real hardware (no obfuscation)")
    print("-" * 60)
    all_results['RH_C0'] = evaluate_dataset("RH_C0 (Raw Real Hardware)", X, y)

    print("\n" + "-" * 60)
    print("  RH_C1: NLFSR-Shield applied to real hardware")
    print("-" * 60)
    y_obf = apply_nlfsr_to_responses(y)[32:]
    all_results['RH_C1'] = evaluate_dataset(
        "RH_C1 (NLFSR-Obfuscated Real Hardware)", X[32:], y_obf)

    print("\n" + "-" * 60)
    print("  RH_MASKED: Good bits only [1,5,6,9,13]")
    print("-" * 60)
    df_full = pd.read_csv(os.path.join(DATA_DIR, 'crp_real_fpga_full.csv'))
    df_full['bit_pos'] = np.tile(np.arange(16),
                                 len(df_full) // 16 + 1)[:len(df_full)]
    df_good = df_full[df_full['bit_pos'].isin(GOOD_BITS)] \
        .sample(n=100_000, random_state=42).reset_index(drop=True)
    all_results['RH_MASKED'] = evaluate_dataset(
        "RH_MASKED (Good Bits Only)",
        df_good[CH_COLS].values.astype(np.float32),
        df_good['response'].values.astype(np.int32))

    print("\n" + "=" * 60)
    print("  FINAL TABLE - Real Hardware ML Results")
    print("=" * 60)
    print(f"  {'Model':<14} {'RH_C0':>8} {'RH_C1':>8} {'RH_MASKED':>10}")
    print("  " + "-" * 44)
    for model in ['LR', 'DNN_adam', 'XGBoost']:
        print(f"  {model:<14} {all_results['RH_C0'][model]['mean']:>7.2f}%"
              f" {all_results['RH_C1'][model]['mean']:>7.2f}%"
              f" {all_results['RH_MASKED'][model]['mean']:>9.2f}%")
    print("=" * 60)

    out_path = os.path.join(OUTPUT_DIR, 'measured_dataset_attack.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
