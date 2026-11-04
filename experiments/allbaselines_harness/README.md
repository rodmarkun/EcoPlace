# All-baselines development comparison harness

This reusable package runs every plan-§6.2 reusable-policy comparator on one named,
quarantined development scenario and one exact normalized evaluator-work budget.
Same-DSL GP, classic GPHH, and random-AST are work-matched adaptive arms. Frozen PPO
and all six hand-written policies are evaluation-only references and receive no
adaptation budget. The Amendment-A1 Codex subscription arm is an explicit opt-in slot.

Non-LLM development run (safe default):

```bash
experiments/allbaselines_harness/run.sh \
  dev-medium-cluster \
  evolution/task/config_dev_medium.json \
  58.8 \
  800000011
```

Population-parity development run (also non-LLM):

```bash
experiments/allbaselines_harness/run.sh \
  dev-medium-cluster \
  evolution/task/config_dev_medium.json \
  58.8 \
  800000011 \
  --seed-archive 2
```

`--seed-archive K` creates one common DSL initialization archive from the six
hand-written PlacementScript reference implementations plus `K` deterministically
seeded random-AST mutants. The harness fails closed unless `6 + K` exactly equals the
configured classic-GPHH population (currently eight, hence `K=2`) and the work budget
can fit every initialization at the maximum L0+L1+L2 charge. The same-DSL GP, LLM slot,
and Random-AST arm evaluate those archive members inside their adaptive work ledgers;
the report verifies each charge with `EvaluatorWorkBudget.work_for_trace` against
classic GPHH generation 0. Initialization therefore consumes part of the common 58.8
budget rather than appearing as free pre-search scoring.

Full run, including live Codex calls (for the supervisor only):

```bash
experiments/allbaselines_harness/run.sh \
  dev-medium-cluster \
  evolution/task/config_dev_medium.json \
  58.8 \
  800000011 \
  --seed-archive 2 \
  --include-llm
```

Lean deep-search run with three diverse proposals per LLM call:

```bash
experiments/allbaselines_harness/run.sh \
  dev-medium-cluster \
  evolution/task/config_dev_medium.json \
  294.0 \
  800000011 \
  --profile lean-deep \
  --proposals-per-call 3 \
  --include-llm
```

`lean-deep` enables the shared behavioral-novelty L2 gate and disables unsupported
proposal interleaves and the bundled enhanced guidance. `--proposals-per-call` affects
only the LLM arm. Its default is 1, which retains the existing single-proposal path.

Arguments are `SCENARIO TASK_CONFIG NORMALIZED_WORK_BUDGET ROOT_SEED`, plus optional
`--profile NAME`, `--engine {codex,openrouter}`, `--engine-config PATH`,
`--proposals-per-call K`, `--seed-archive K`, `--runs-root DIR`, and `--include-llm`.
The engine defaults to Codex; each engine's committed config is the default. Single-seed, population-
parity, non-LLM, and all-arms modes use different deterministic run directories, so a
sealed report cannot be mutated into another mode. `--runs-root` must resolve inside
the repository and makes the artifact root explicit. Runs resume adaptive archives
and refuse any changed input or sealed output.

For Codex, preflight checks the runner-facing durable usage ledger and proves both
Amendment-A1 confirmatory refusals are active. For OpenRouter, it checks the pinned model,
config digest, and price table. An enabled LLM run performs one fail-closed live marker
probe and journals its volatile evidence outside the sealed payload. Each run emits a sealed
`report.json` and `REPORT.md` under `runs/` with exact Phase-3 sort keys, normalized
energy, existing best/final incumbents, initialization best, absolute best, search
delta, best-found generation/proposal, the derived `gphh_gen0` random-population arm,
plateau histograms, improvements against the common initial policy and every hand-
written reference, work/call accounting, and mechanical health.

Arm results use compact v2 evidence: exact policy/champion identities, fitness and sort
keys, per-instance energy/accounting scalars, ladder and plateau summaries, and SHA-256
references to hash-chained campaign archives. Full simulator episode and decision
ledgers are not copied into arm JSON. `harness_config.json` fails closed above 10 MiB
for a new arm result, 1 GiB for any artifact, or 4 GiB for one run directory; adaptive
subprocesses are monitored while they run.

Legacy seals compressed by the supervisor remain directly readable. Both the normal
loader and the constant-memory verifier resolve `result.json` to `result.json.gz` when
needed; neither writes into the run directory:

```bash
.venv/bin/python -m experiments.allbaselines_harness.verify \
  experiments/allbaselines_harness/runs/RUN/arms/classic_gphh/result.json
```

Passing a run directory verifies every recognized sealed input, result, preflight, and
report beneath it. The verifier streams multi-GB legacy JSON through gzip and checks the
original canonical payload SHA-256 without materializing or manually decompressing it.

Classic GPHH remains a native DEAP policy and is not PlacementScript fuel-metered.
This known compute-accounting asymmetry is recorded in every report and is not changed
by population parity.

Interpretation boundary: quarantined development comparison only; never acceptance,
P1/P2, held-out, or confirmatory evidence.
