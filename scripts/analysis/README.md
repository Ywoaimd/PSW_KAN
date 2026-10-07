# Ablation and computational analysis

These additions contain model code, configurations and executable scripts only. No experiment results, logs, datasets or checkpoints are distributed.

Validated runtime: Python 3.9.23, PyTorch 2.5.1+cu121, NumPy 1.22.4 (the tkan environment). Use the repository data layout described in the main README. ETT paths in the public configuration have been adapted to that layout; training hyperparameters are unchanged.

From the repository root:

```bash
python scripts/analysis/run_ablation.py --repo . --dataset ETTh1 --horizon 96 --variant full --seed 42 --output ablation_runs
# Add --execute to train; the default command only prints the configuration.
EXPERIMENT_PYTHON=python bash scripts/analysis/run_matched36.sh "$PWD" "$PWD/ablation_runs"
python scripts/analysis/profile_complete.py --repo . --output complexity_profile.json
```

The queue trains Electricity and ETTh1, horizons 96/336, full/no_patch/no_reswave, seeds 42/123/456. Existing logs are never overwritten. The profiler covers six datasets and four horizons for full PSW-KAN and TimeKAN. Batch size is one, input length is 96, and multiply-add is two FLOPs. FFT and nonlinear-function costs are explicit arithmetic approximations in the script. It reports unsupported operators instead of silently claiming they are counted. These are forward estimates, not training costs or hardware instruction counts.
