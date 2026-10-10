# Supplementary Code (version 3, second revision)

Supplementary material for *"Design and Simulation of a Lightweight Nonlinear
Feedback Shift Register Obfuscation Layer Against Machine Learning Attacks on
Configurable Ring Oscillator PUFs"*.

Requirements: Python 3.10+ with `numpy`, `scipy`, `pandas`, `scikit-learn`,
`xgboost` and `tensorflow`; Icarus Verilog 12 for the RTL checks; Xilinx Vivado
(free edition) for implementation; any C99 compiler for the period programs.

## What changed in version 3

* **Independent virtual devices.** Device *d* is drawn from its own PCG64 stream
  (`SeedSequence(0x5EED20260BADF00E, spawn_key=(d,))`) with Gaussian process
  variation. The chained 16-bit LCG of versions 1 and 2 is kept only in
  `model/legacy_lcg.py` for the comparison study; it is not full-period
  (4258 is not divisible by 4) and chained seeds made consecutive devices
  shifted copies of each other.
* **Gate-level RTL with real ring oscillators** (`rtl_hw/`) and a one-command
  Vivado flow (`vivado_hw/`). The earlier implementation results were obtained
  from the behavioural model, which contains no oscillators.
* **NLFSR v2.1:** new `advance` input, so that the register steps exactly once
  per PUF response, as in the Python model.
* Two new studies in `experiments/`.

## model/

| File | Purpose |
|---|---|
| `puf_core.py` | The shared behavioural model (v3.0). `python model/puf_core.py` runs its self-test. |
| `legacy_lcg.py` | The generator of versions 1 and 2, for comparison only. |

Expected self-test output:

    Uniformity (30 devices) : 49.87%
    Uniformity (device 0)   : 48.44%
    Uniqueness              : 49.27%
    Neighbour correlation   : +0.0144
    Effective challenge bits: 16 of 16 across devices; per device mean 14.23 (min 8, max 16)
    NLFSR bit-31 collisions : 0 of 100000 (must be 0)
    All checks passed.

A challenge bit can be ineffective on an individual device when the oscillator
pair it configures has a base-frequency gap larger than the total configurable
delay (a stable pair). Every bit is effective in the model.

## generation/ and attacks/

    python generation/generate_datasets.py          # C0: 100K / 500K / 1M / 3M
    python generation/generate_c1_c2_datasets.py    # C1 and C2
    python attacks/ablation_study.py                # Table XI (Protocol A)
    python attacks/measured_dataset_attack.py     # Table XII (Protocol A)
    python attacks/ml_regen_g4.py                   # Table VIII, single-device C0 at 100K (Protocol B, seeds 42/123/456)
    python attacks/ml_regen3.py                     # all other entries of Tables VIII and X (Protocol B, seed 42)

`ml_regen_g4.py` produced the single-device C0 values at 100K (three seeds, DNNs
trained for three epochs); its later stages were too slow on the available
hardware and were not used. `ml_regen3.py` takes those values as fixed and
produces every other entry of Tables VIII and X (seed 42, DNNs trained for two
epochs); it writes `attacks/ml_regen_g4_results.json`, the exact values printed in
the paper. Section 4.6 of the paper describes both protocols.

## rtl/ (behavioural model, used to verify the datasets)

| File | Purpose |
|---|---|
| `cropuf_sim_model.v` | Behavioural CRO-PUF (v3.0), loads device parameters with `$readmemh`. |
| `nlfsr_core.v` | 32-bit NLFSR, MNL taps 30, 15, 7 (v2.1). |
| `tb_equivalence_*.v` | Dump RTL outputs for comparison with Python. |

    python tools/export_device_hex.py 0
    iverilog -g2005 -o sim_cropuf rtl/cropuf_sim_model.v rtl/tb_equivalence_cropuf.v && vvp sim_cropuf
    iverilog -o sim_nlfsr rtl/nlfsr_core.v rtl/tb_equivalence_nlfsr.v && vvp sim_nlfsr
    python tools/check_equivalence.py

Result: all 65,535 challenges of one LFSR period, 0 mismatches; NLFSR 3,000
bits, 0 mismatches. Stimulus is driven on the negative clock edge.

## rtl_hw/ (gate-level design with real ring oscillators)

| File | Purpose |
|---|---|
| `ciu.v` | Configurable Inversion Unit: XOR, XNOR and MUX as separate DONT_TOUCH LUTs. |
| `ro_cell.v` | Five-stage ring: enable NAND, three CIUs, inverter. |
| `cro_puf_core.v` | 16 rings, pair selection, two edge counters, 65,536-cycle window, FSM. |
| `keccak_f200_layer.v` | C2 output stage (bit-exact with the Python model). |
| `top_c0.v`, `top_c1.v`, `top_c2.v` | Synthesis tops of the three configurations. |
| `ro_delay_table.v` | Simulation only: gate delays of one virtual device. |
| `tb_ro_timing.v`, `tb_c1_timing.v`, `tb_keccak_layer.v` | Testbenches. |

Timing simulation (the rings really oscillate; gate delays from device 0):

    python tools/export_ro_delays.py 0
    iverilog -g2005 -DRO_TIMING_SIM -o sim_ro rtl_hw/ro_delay_table.v rtl_hw/ciu.v \
        rtl_hw/ro_cell.v rtl_hw/cro_puf_core.v rtl_hw/tb_ro_timing.v && vvp sim_ro
    python tools/check_ro_timing.py 0

Result (`experiments/ro_timing/`): 2,000 challenges, agreement 99.90%; both
disagreements are equal edge counts where the model margin is at most 2 units.
`tb_c1_timing.v` + `tools/check_c1_hw.py`: 400 C1 outputs, 0 mismatches.
`tb_keccak_layer.v` + `tools/check_keccak.py`: 640 bits, 0 mismatches.

## vivado_hw/ (implementation, no board needed)

Start Vivado without opening a project; in the Tcl Console:

    cd <this folder>/vivado_hw
    source run_all.tcl

Synthesises, places, routes and writes a bitstream for C0, C1 and C2 on
xc7a35tcpg236-1 with a 10 ns clock. Reports are written to `vivado_hw/reports/`
and the key numbers to `reports/summary.txt` (Table XIII).

## experiments/

| Script | Question |
|---|---|
| `independent_seed_study.py` | Do the 1M results come from the device generator? (Section 5.4.1, Table IX) |
| `physical_robustness_study.py` | Do the conclusions survive noise, layout asymmetry and shared variation? (Section 5.10, Table XV) |
| `grouped_split_study.py` | Random vs challenge-grouped vs class-grouped split, per device (Section 5.9, Table XIV) |
| `sequence_models.py` | LSTM and GRU on the ordered C1 stream (Section 5.11, Table XVI). Needs PyTorch. |
| `design_aware_analysis.py` | Structural properties that bound a design-aware adversary: 416 effective challenge classes, the public keystream recurrence, and the 32-bit seed entropy (Section 6.5). |

`verification/nlfsr_period_seeds.c` gives the NLFSR period for each device seed
(Table I); build with `gcc -O3 -o nlfsr_period_seeds verification/nlfsr_period_seeds.c`
and run on `experiments/device_seeds.txt`.

## verification/

`nist_sp800_22.py`, `run_nist.py` (Tables III and IV); `nlfsr_period.c`,
`nlfsr_fix.c`, `verify_period*.py` (NLFSR periods).

## figures/

`make_figures_4_to_8.py` draws Figures 4 to 8 from the per-bit uniformity of the
measured dataset and the values in Tables X, XI, XII and XIII (600-dpi PNG and
vector PDF):

    python figures/make_figures_4_to_8.py
