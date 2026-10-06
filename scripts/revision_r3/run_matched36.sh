#!/usr/bin/env bash
set -euo pipefail
# Run manually only after checking data paths. This script starts 36 trainings (12 matched full references plus 24 ablations).
repo="${1:?Usage: bash run_missing24.sh /absolute/path/to/TimeKAN [output_directory]}"
output="${2:-$PWD/r3_runs}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$output/logs"
for dataset in Electricity ETTh1; do
  for horizon in 96 336; do
    for variant in full no_patch no_reswave; do
      for seed in 42 123 456; do
        logfile="$output/logs/${dataset}_${horizon}_${variant}_seed${seed}.log"
        if [[ -e "$logfile" ]]; then
          echo "Refusing to overwrite $logfile" >&2
          exit 1
        fi
        "${R3_PYTHON:-python}" "$script_dir/run_ablation.py" --repo "$repo" --dataset "$dataset" --horizon "$horizon" --variant "$variant" --seed "$seed" --output "$output" --execute > "$logfile" 2>&1
      done
    done
  done
done
