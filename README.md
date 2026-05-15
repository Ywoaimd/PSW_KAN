# PSW-KAN

This repository contains the implementation of PSW-KAN for long-term multivariate time series forecasting.

PSW-KAN is a lightweight architectural enhancement of TimeKAN. It keeps the decomposition-oriented KAN backbone and adds residual-gated channel recalibration, ResWave high-frequency refinement, and a PatchLite auxiliary branch.

## Requirements

```bash
pip install -r requirements.txt
```

The experiments were run with PyTorch 1.13.1. CUDA availability depends on the local PyTorch installation.

## Data

Download the public LTSF benchmark datasets and place them under `dataset/`:

```text
dataset/
  ETTh1/ETTh1.csv
  ETTh2/ETTh2.csv
  ETTm1/ETTm1.csv
  ETTm2/ETTm2.csv
  weather/weather.csv
  electricity/electricity.csv
```

The dataset files are not included in this repository.

## Training

Run one dataset/horizon script from the repository root, for example:

```bash
bash scripts/Electricity/Electricity_96.sh
```

Run all six datasets and four horizons:

```bash
bash scripts/run_all.sh
```

The scripts use `--model PSW_KAN`. Main outputs are written to `result_long_term_forecast.txt`, `txt_results/`, `results/`, `test_results/`, and `checkpoints/`; these runtime artifacts are ignored by Git.

## Structure

```text
run.py                  # training and evaluation entry point
data_provider/          # LTSF dataset loaders
exp/                    # experiment loop
layers/                 # TimeKAN backbone layers
models/PSW_KAN.py       # PSW-KAN model
modules/AdpWavelet.py   # ResWave-related block
modules/PatchLite/      # lightweight patch branch
scripts/                # reproduced experiment scripts
```

## Acknowledgement

This implementation builds on the TimeKAN codebase and common LTSF experiment infrastructure.
