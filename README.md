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

*To be filled in: dataset generation, then attacks, then analysis, then figures.*

## Data

The measured third-party dataset (Abulibdeh et al.) is available from IEEE DataPort: https://doi.org/10.21227/6ssx-ft56. It is not redistributed here.

## Citation

See `CITATION.cff`.

## License

Code and RTL: Apache-2.0 (see `LICENSE`).
