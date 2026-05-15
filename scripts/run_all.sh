#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] Please run with bash: bash scripts/run_all.sh [--dry-run] [--keep-going]" >&2
  exit 1
fi
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

if command -v git >/dev/null 2>&1; then
  if git -C "$repo_root" rev-parse --show-toplevel >/dev/null 2>&1; then
    repo_root="$(git -C "$repo_root" rev-parse --show-toplevel)"
  fi
fi

declare -a datasets=(ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity)
declare -a horizons=(96 192 336 720)

DRY_RUN=false
KEEP_GOING=false

usage() {
  echo "Usage: $0 [--dry-run] [--keep-going]" >&2
  echo "  --dry-run     Only print the commands, do not execute" >&2
  echo "  --keep-going  Continue on error" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --keep-going)
      KEEP_GOING=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

shopt -s nullglob

echo "[INFO] scripts root: $script_dir"
echo "[INFO] repo root:    $repo_root"

time_log="$repo_root/txt_results/run_all_time.txt"
if ! $DRY_RUN; then
  mkdir -p "$(dirname "$time_log")"
fi

total=0
skipped=0
failed=0
ran=0

for ds in "${datasets[@]}"; do
  ds_dir="$script_dir/$ds"
  if [[ ! -d "$ds_dir" ]]; then
    echo "[WARN] Dataset dir not found: $ds_dir - skipping"
    continue
  fi

  ds_start_ts=$(date +%s)
  ds_failed=0
  ds_ran=0
  ds_skipped=0
  printf "\n[DATASET] %s\n" "$ds"

  for h in "${horizons[@]}"; do
    matches=("$ds_dir"/*_"$h".sh)
    if [[ ${#matches[@]} -eq 0 ]]; then
      echo "[SKIP] $ds $h - no script found matching '*_${h}.sh' in $ds_dir"
      ((skipped+=1))
      ((ds_skipped+=1))
      ((total+=1))
      continue
    fi

    selected="${matches[0]}"
    for cand in "${matches[@]}"; do
      base="$(basename "$cand")"
      if [[ "${base,,}" == "${ds,,}_${h}.sh" ]]; then
        selected="$cand"
        break
      fi
    done

    echo "[RUN] $ds $h -> $selected"
    ((total+=1))

    if $DRY_RUN; then
      continue
    fi

    chmod +x "$selected" || true
    if ! (cd "$repo_root" && bash "$selected"); then
      echo "[FAIL] $selected"
      ((failed+=1))
      ((ds_failed+=1))
      if ! $KEEP_GOING; then
        echo "[EXIT] Stopping due to error. Use --keep-going to continue."
        exit 1
      fi
    else
      ((ran+=1))
      ((ds_ran+=1))
    fi
  done

  ds_end_ts=$(date +%s)
  ds_elapsed=$((ds_end_ts - ds_start_ts))
  if ! $DRY_RUN; then
    if [[ $ds_failed -gt 0 ]]; then
      ds_status="failed"
    else
      ds_status="ok"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" "$ds" "$ds_elapsed" "$ds_status" "$ds_ran" "$ds_skipped" >> "$time_log"
  fi
done

printf "\n[SUMMARY] total:%d ran:%d skipped:%d failed:%d\n" "$total" "$ran" "$skipped" "$failed"
if [[ $failed -gt 0 ]]; then
  exit 2
fi
