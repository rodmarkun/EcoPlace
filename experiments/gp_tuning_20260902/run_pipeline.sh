#!/usr/bin/env bash
# Sequence mix56 -> budget224 -> budget112, rescoring after each phase.
# Zero LLM calls. Designed to survive a killed parent via `setsid nohup`.
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

HERE="experiments/gp_tuning_20260902"
LOG_ROOT="${LOG_ROOT:-$HERE/logs}"
RUNS_ROOT="${RUNS_ROOT:-$HERE/runs}"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$LOG_ROOT"

count_reports() {
  local phase="$1"
  "$PYTHON" - "$RUNS_ROOT" "$phase" <<'PY'
from pathlib import Path
import sys
runs = Path(sys.argv[1])
phase = sys.argv[2]
n = 0
if not runs.is_dir():
    print(0)
    raise SystemExit
for path in runs.glob("*/report.json"):
    name = path.parent.name
    if phase == "mix56":
        if "work-56000000" in name and any(
            name.endswith(f"--gp-mix-{mix}")
            for mix in ("numeric-heavy", "subtree-heavy", "crossover-heavy")
        ):
            n += 1
    elif phase == "budget224":
        if "work-224000000" in name and name.endswith("--gp-mix-current"):
            n += 1
    elif phase == "budget112":
        if "work-112000000" in name and name.endswith("--gp-mix-current"):
            n += 1
print(n)
PY
}

phase_expected() {
  case "$1" in
    mix56) printf '24' ;;
    budget224) printf '8' ;;
    budget112) printf '8' ;;
    *) return 1 ;;
  esac
}

run_phase() {
  local phase="$1"
  local expected have
  expected=$(phase_expected "$phase")
  have=$(count_reports "$phase")
  echo "phase ${phase}: ${have}/${expected} report.json present"
  echo $$ > "$LOG_ROOT/grid-${phase}.pid"
  if [[ "$have" -lt "$expected" ]]; then
    echo "starting ${phase} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    bash "$HERE/run_grid.sh" "$phase"
  else
    echo "skip ${phase}: already complete"
  fi
  have=$(count_reports "$phase")
  echo "phase ${phase} reports now ${have}/${expected}"
  echo "rescoring after ${phase} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PYTHON" "$HERE/analyze.py" --rescore
  echo "${phase} analysis finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

echo "pipeline start $$ at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo $$ > "$LOG_ROOT/pipeline.pid"
run_phase mix56
run_phase budget224
run_phase budget112
rm -f "$LOG_ROOT/grid-mix56.pid" "$LOG_ROOT/grid-budget224.pid" "$LOG_ROOT/grid-budget112.pid"
echo "pipeline finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "done" > "$LOG_ROOT/pipeline.done"
