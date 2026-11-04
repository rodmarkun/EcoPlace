#!/usr/bin/env bash
# GP operator-mix grid and budget sweep. Zero LLM calls. Concurrency <= 3.
#
# Phase order:
#   mix56     three non-default mixes at 56.0 (current@56 is reused from runs_final)
#   budget224 current mix at 224.0 (headline 4x comparison; run before 112.0)
#   budget112 current mix at 112.0
#   all       mix56, then budget224, then budget112
#
# Concurrency uses kill -0 on launched PIDs so this script still caps at 3 under
# `setsid nohup bash` (non-interactive bash has no job control, so `jobs -rp` is 0).
# Roots that already have report.json are skipped (resume-safe).
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$repo_root"

PHASE="${1:-all}"
CONCURRENCY="${CONCURRENCY:-3}"
RUNS_ROOT="${RUNS_ROOT:-experiments/gp_tuning_20260902/runs}"
LOG_ROOT="${LOG_ROOT:-experiments/gp_tuning_20260902/logs}"
TASK_CONFIG="evolution/task/config_dev_continuum_tight_calibrated.json"
INITIAL="evolution/task/initial_tight.py"
ENGINE_CONFIG="evolution/openrouter_engine_glm53.json"
PROFILE="deep-search-feedback"
CONFIG_DIR="experiments/gp_tuning_20260902/configs"
ROOTS=(800000801 800000802 800000803 800000804 800000805 800000806 800000807 800000808)
MIXES_56=(numeric-heavy subtree-heavy crossover-heavy)
LIVE_PIDS=()
LAUNCHED_PIDS=()

mkdir -p "$RUNS_ROOT" "$LOG_ROOT"

microwork() {
  awk -v b="$1" 'BEGIN { printf "%d", int(b * 1000000 + 0.5) }'
}

run_dir_for() {
  local budget="$1"
  local mix="$2"
  local root="$3"
  printf '%s/dev-continuum-tight--seed-%s--work-%s--non-llm--%s--engine-openrouter--seed-archive-5--batch-3--gp-mix-%s' \
    "$RUNS_ROOT" "$root" "$(microwork "$budget")" "$PROFILE" "$mix"
}

prune_live() {
  local alive=()
  local pid
  for pid in "${LIVE_PIDS[@]+"${LIVE_PIDS[@]}"}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  if [[ ${#alive[@]} -gt 0 ]]; then
    LIVE_PIDS=("${alive[@]}")
  else
    LIVE_PIDS=()
  fi
}

wait_for_slot() {
  while true; do
    prune_live
    if [[ ${#LIVE_PIDS[@]} -lt ${CONCURRENCY} ]]; then
      return
    fi
    sleep 20
  done
}

launch_one() {
  local budget="$1"
  local mix="$2"
  local root="$3"
  local dest log
  dest=$(run_dir_for "$budget" "$mix" "$root")
  early="${dest/\/runs\//\/runs_budget224_early\/}"
  if [[ -f "${dest}/report.json" || -f "${early}/report.json" ]]; then
    echo "skip complete mix=${mix} budget=${budget} root=${root}"
    return 0
  fi
  log="$LOG_ROOT/gp-${mix}-w${budget}-root-${root}.log"
  echo "launching mix=${mix} budget=${budget} root=${root} -> ${log}"
  nohup experiments/allbaselines_harness/run.sh \
    dev-continuum-tight \
    "$TASK_CONFIG" \
    "$budget" \
    "$root" \
    --profile "$PROFILE" \
    --engine openrouter \
    --engine-config "$ENGINE_CONFIG" \
    --proposals-per-call 3 \
    --seed-archive 5 \
    --initial "$INITIAL" \
    --runs-root "$RUNS_ROOT" \
    --gp-config "$CONFIG_DIR/${mix}.json" \
    >>"$log" 2>&1 &
  LIVE_PIDS+=($!)
  LAUNCHED_PIDS+=($!)
}

run_cells() {
  local budget="$1"
  shift
  local mixes=("$@")
  local status=0
  local pid
  LIVE_PIDS=()
  LAUNCHED_PIDS=()
  for mix in "${mixes[@]}"; do
    for root in "${ROOTS[@]}"; do
      wait_for_slot
      launch_one "$budget" "$mix" "$root"
    done
  done
  for pid in "${LAUNCHED_PIDS[@]+"${LAUNCHED_PIDS[@]}"}"; do
    if ! wait "$pid"; then status=1; fi
  done
  return "$status"
}

status=0
case "$PHASE" in
  mix56)
    run_cells 56.0 "${MIXES_56[@]}" || status=1
    ;;
  budget224)
    run_cells 224.0 current || status=1
    ;;
  budget112)
    run_cells 112.0 current || status=1
    ;;
  all)
    run_cells 56.0 "${MIXES_56[@]}" || status=1
    run_cells 224.0 current || status=1
    run_cells 112.0 current || status=1
    ;;
  *)
    echo "usage: $0 {mix56|budget224|budget112|all}" >&2
    exit 64
    ;;
esac

echo "phase ${PHASE} finished (status ${status})"
exit "$status"
