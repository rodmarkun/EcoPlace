#!/usr/bin/env bash
set -euo pipefail

harness_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$harness_dir/../.." && pwd)

if [[ $# -lt 4 ]]; then
  echo "usage: $0 SCENARIO TASK_CONFIG NORMALIZED_WORK_BUDGET ROOT_SEED [--profile NAME] [--engine {codex,openrouter}] [--engine-config PATH] [--proposals-per-call K] [--seed-archive K] [--runs-root DIR] [--gp-config PATH] [--include-llm]" >&2
  exit 64
fi

cd "$repo_root"
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python -m experiments.allbaselines_harness.runner "$@"
