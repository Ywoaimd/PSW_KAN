#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  echo "[ERROR] Please run with bash: bash scripts/run_ecl_full.sh [--dry-run] [--keep-going]" >&2
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

declare -a horizons=(
  96 192 336 720
)

DRY_RUN=false
KEEP_GOING=false

usage() {
  echo "Usage: $0 [--dry-run] [--keep-going]" >&2
  echo "  --dry-run     Only print the command, do not execute" >&2
  echo "  --keep-going  Continue on error (do not exit immediately)" >&2
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

echo "[INFO] scripts root: $script_dir"
echo "[INFO] repo root:    $repo_root"
echo "[INFO] dataset:      Electricity"
echo "[INFO] horizons:     ${horizons[*]}"

total=0
ran=0
failed=0

for h in "${horizons[@]}"; do
  target="$script_dir/Electricity/Electricity_${h}.sh"
  ((total+=1))
  if [[ ! -f "$target" ]]; then
    echo "[FAIL] Script not found: $target"
    ((failed+=1))
    if ! $KEEP_GOING; then
      echo "[EXIT] Stopping due to error. Use --keep-going to continue on errors."
      exit 1
    fi
    continue
  fi

  echo "[RUN] ECL ${h} -> $target"
  if $DRY_RUN; then
    continue
  fi

  chmod +x "$target" || true
  if ! ( cd "$repo_root" && bash "$target" ); then
    echo "[FAIL] $target"
    ((failed+=1))
    if ! $KEEP_GOING; then
      echo "[EXIT] Stopping due to error. Use --keep-going to continue on errors."
      exit 1
    fi
  else
    ((ran+=1))
  fi
done

printf "\n[SUMMARY] total:%d ran:%d failed:%d\n" "$total" "$ran" "$failed"
if [[ $failed -gt 0 ]]; then
  exit 2
fi
