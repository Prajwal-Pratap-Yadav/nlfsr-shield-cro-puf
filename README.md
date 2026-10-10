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
| `rtl/` | Verilog RTL (gate-level and behavioural) and testbenches |
| `vivado/` | Tcl scripts, constraints, post-implementation reports |
| `model/` | Python behavioural device model |
| `datasets/` | CRP generation scripts and SHA-256 manifest |
| `attacks/` | ML attack scripts |
| `analysis/` | PUF metrics, NIST tests, ablation, robustness, power-analysis simulation |
| `figures/` | Figure-generation scripts and final figures |
| `results/` | Result files for the reported experiments |
| `docs/` | Manuscript section to script mapping and reproduction notes |

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
