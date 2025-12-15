# STeP Simulator Hardware Verification

This artifact validates the symbolic STeP simulator against a cycle-accurate HDL model for a SwiGLU layer running on a tiled spatial accelerator. The workflow mirrors the paper: build the STeP graph, retile it to the hardware’s 16×16 compute tiles, generate Bluespec, simulate with a Ramulator2-backed HBM2 model, and compare cycle counts and off-chip traffic.

## What’s here
- `src/`: Rust STeP front-end and compiler passes. Defines the graph IR, tiling/downtiling logic, dot export, and codegen utilities used by the paper’s experiments.
  - `hwsim/graph` and `passes/`: symbolic graph construction plus passes like the downtiler and conversion to static Bluespec.
  - `test/`: paper-oriented tests; `run_dse_sweep` builds the validation graphs, exercises the HDL, and writes `dse_results.csv`.
- `bluespec/`: cycle-accurate Bluespec SystemVerilog model of the tiled accelerator, plus BDPI hooks into Ramulator2 for HBM timing.
  - `ramulator2/`: vendored Ramulator2 source.
- Outputs `validation.tex` / `validation.pdf` (after running): LaTeX for the validation figure; `dse_results.csv` and `sweep_results_rescored.csv` hold the sweep results used in the plot.
- `run_dse_and_figure.sh`: helper scripts to regenerate the sweep data and figure.

## Typical usage
- Regenerate sweep data and the validation figure:
  - Run STeP Simulator to provide results for `step_reference.csv`.
  - Requirements (already included in the submitted docker): Rust/cargo, `cmake` + `make` (for Ramulator2), and `latexmk` for the PDF, and Bluespec `bsc`.
  - Command: `./run_dse_and_figure.sh`
    - Builds Ramulator2 if missing, runs `cargo test run_dse_sweep -- --nocapture` to produce `dse_results.csv`, then compiles the LaTeX figure.
    - The figure will be stored in `validation.pdf`.

The resulting data shows the close agreement between the STeP simulator and the HDL (Pearson ≈ 0.99 on cycle counts) while also correlating off-chip traffic measured in the symbolic front-end with the Ramulator2-backed HDL runs.

