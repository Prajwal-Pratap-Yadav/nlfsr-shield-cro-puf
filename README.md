# NLFSR-Shield for Configurable Ring Oscillator PUFs

A lightweight 32-bit nonlinear feedback shift register (NLFSR) obfuscation layer that removes the machine-learning modelling attacks evaluated against 16-bit configurable ring oscillator PUFs (CRO-PUFs). This repository holds the code accompanying the paper:

> **Design and Simulation of a Lightweight Nonlinear Feedback Shift Register Obfuscation Layer Against Machine Learning Attacks on Configurable Ring Oscillator PUFs**
> P. P. Yadav, E. Pramanick, B. Acharya, A. K. Dedefo. *(journal / DOI to be added)*

## Overview

The 16-bit challenge space of a conventional CRO-PUF collapses, under the pair-selection rule, to only a few hundred effective input classes, so capacity-sufficient ML models reach up to 100% prediction accuracy at 100,000 CRPs on a single device. NLFSR-Shield XORs the raw response with a keystream from a 32-bit NLFSR (maximal-length linear taps plus a 3-input maximally nonlinear combining function), bringing attack accuracy to about 50%.

Three configurations are implemented:

| Config | Description |
|--------|-------------|
| C0 | Unprotected CRO-PUF baseline |
| C1 | NLFSR-Shield (proposed) |
| C2 | Keccak-f[200] reference |

## Scope

This is a **design-and-simulation study**. The RTL was verified in event-driven timing simulation and implemented in Vivado for the Artix-7 `xc7a35tcpg236-1` up to a bitstream, but nothing was programmed on a physical board and every CRP used in the evaluation comes from a behavioural model. NLFSR-Shield is an obfuscation layer, not a cipher.

## Repository layout

| Folder | Contents |
|--------|----------|
| `rtl/` | Behavioural simulation models (`cropuf_sim_model.v`, `nlfsr_core.v`) and equivalence testbenches |
| `rtl_hw/` | Gate-level RTL: CIU, ring-oscillator cells, CRO-PUF core, `top_c0` / `top_c1` / `top_c2`, Keccak-f[200] layer, timing testbenches |
| `vivado_hw/` | Vivado flow (`run_all.tcl`), constraints (`nlfsr_shield.xdc`), post-implementation reports for C0/C1/C2 |
| `model/` | Python device model (`puf_core.py`) and the earlier LCG generator (`legacy_lcg.py`) kept for comparison |
| `generation/` | CRP dataset generation for C0, C1 and C2 |
| `attacks/` | ML attack scripts and ablation study |
| `experiments/` | Grouped-split, independent-seed, physical-robustness, sequence-model (LSTM/GRU), design-aware and RO-timing studies |
| `verification/` | NLFSR period verification (C and Python) and NIST SP 800-22 tests |
| `tools/` | RTL/model equivalence checks and export helpers |
| `figures/` | Figure-generation script |
| `docs/` | Original supplementary README and reproduction notes |

## Requirements

- Python 3.11 (`pip install -r requirements.txt`)
- Vivado Design Suite 2025.2 (target: `xc7a35tcpg236-1`)
- Icarus Verilog (and/or Vivado XSim)

## Reproducing the results

All commands run from the repository root.

**Setup**

```bash
pip install -r requirements.txt   # PyTorch is needed for experiments/sequence_models.py
```

Icarus Verilog 12 is needed for the RTL checks, Vivado for implementation, and any C99 compiler for the period programs.

**1. Sanity checks**

```bash
python model/puf_core.py                         # device-model self-test
python tools/export_device_hex.py 0
iverilog -g2005 -o sim_cropuf rtl/cropuf_sim_model.v rtl/tb_equivalence_cropuf.v && vvp sim_cropuf
iverilog -o sim_nlfsr rtl/nlfsr_core.v rtl/tb_equivalence_nlfsr.v && vvp sim_nlfsr
python tools/check_equivalence.py                # expect 0 mismatches
```

**2. Generate CRP datasets**

```bash
python generation/generate_datasets.py           # C0: 100K / 500K / 1M / 3M
python generation/generate_c1_c2_datasets.py     # C1 and C2
```

**3. ML attacks and studies**

| Manuscript item | Command |
|---|---|
| Tables VIII, X (Protocol B) | `python attacks/ml_regen_g4.py` then `python attacks/ml_regen3.py` |
| Table IX | `python experiments/independent_seed_study.py` |
| Table XI (ablation) | `python attacks/ablation_study.py` |
| Table XII (measured third-party data) | `python attacks/measured_dataset_attack.py` |
| Table XIV | `python experiments/grouped_split_study.py` |
| Table XV | `python experiments/physical_robustness_study.py` |
| Table XVI | `python experiments/sequence_models.py` |
| Sections 4.10, 6.5 | `python experiments/design_aware_analysis.py` |

**4. Verification**

| Manuscript item | Command |
|---|---|
| Table I (NLFSR periods) | `gcc -O3 -o nlfsr_period_seeds verification/nlfsr_period_seeds.c`, run on `experiments/device_seeds.txt` |
| Tables III, IV (NIST SP 800-22) | `python verification/run_nist.py` |
| Section 5.1 (gate-level timing simulation) | see `docs/supplementary_README.md` (`tools/check_ro_timing.py`, `check_c1_hw.py`, `check_keccak.py`) |

**5. FPGA implementation (Table XIII)**

In the Vivado Tcl console (no board needed):

```tcl
cd <repo>/vivado_hw
source run_all.tcl
```

Reports go to `vivado_hw/reports/`; key numbers are in `summary.txt`.

**6. Figures 4 to 8**

```bash
python figures/make_figures_4_to_8.py
```

Figures 1 to 3 are schematics drawn from the RTL and are not generated by code.

**Measured data.** `attacks/measured_dataset_attack.py` and Tables VII and XII use the third-party dataset of Abulibdeh et al. (IEEE DataPort, https://doi.org/10.21227/6ssx-ft56), which must be downloaded separately.

Stored outputs of our runs are in the `*_results.json` files under `attacks/` and `experiments/`.

## Data

The measured third-party dataset (Abulibdeh et al.) is available from IEEE DataPort: https://doi.org/10.21227/6ssx-ft56. It is not redistributed here.

## Citation

See `CITATION.cff`.

## License

Code and RTL: Apache-2.0 (see `LICENSE`).
