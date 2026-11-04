# sim/ — continuum discrete-event simulator

A discrete-event simulator (SimPy 4) of microservice workloads on a cloud–edge cluster, with a
Kubernetes-style scheduling cycle into which PlacementScript policies plug through the native
batch API (`placementscript`).

## Modules

| Module | Contents |
|---|---|
| `spec.py` | Frozen, validated episode specifications with canonical JSON I/O and `config_sha256` |
| `runtime.py` | Cluster truth state: pod lifecycle, node capacity and power states, request records, and a conservation ledger that `verify()` reconciles at the end of every episode |
| `app.py` | Application model: open queueing network over service replicas, fan-out/fan-in call graphs, timeouts and bounded retries, load balancing, cold starts, capacity contention |
| `scheduler.py` | Scheduling-cycle emulator matched to a pinned kube-scheduler (Filter → Score → NormalizeScore → Reserve, asynchronous binding, assumed-pod cache, informer staleness, deterministic tie-break); the policy's `score` runs in the Score step |
| `actuation.py` | Budgeted rebalancer behind a transactional action gateway, HPA, and the node power-state controller |
| `energy/` | Closed modeled energy boundary (node power curves, cloud vCPU-hours, network bytes, kept as separate categories), energy-model bundles from pinned published sources (SPECpower and others), and the lexicographic fitness in `energy/fitness.py` |
| `scenarios/` | Workloads and scenarios: Alibaba-derived and synthetic workloads, topology families, drift, and the sealed development scenarios used by the studies (`dev_tight.py`, `dev_realistic.py`, `dev_medium.py`) |
| `engine.py` | `Simulation` / `run_episode`: wires the components and returns a sealed `EpisodeResult` with SLO, goodput and energy metrics and a replay digest |
| `snapshot.py` | Declarative cluster snapshots and tamper-evident snapshot libraries |
| `l0.py` | L0 analytical evaluator: runs a policy over frozen snapshots through the same Filter/Score pipeline as the emulator, then scores power, latency and constraints in closed form |
| `fixtures/` | Development fixtures (`dev/`) and the separate acceptance fixtures (`acceptance/`) used once for the simulator verification gate (`experiments/scientific_gates_20260920`) |
| `benchmarks/l0_bench.py` | L0 throughput benchmark |

Tests: `tests/test_sim_*.py` and `tests/test_scenario_*.py`.

## Determinism

- Every stochastic entry point takes an explicit seed (see [`docs/SEEDS.md`](../docs/SEEDS.md));
  `Simulation` derives independent generators with `analysis.tracking.spawn_rng`.
- Common random numbers: each environmental draw is keyed by the stable identity of what it is
  drawn for (request, call edge and attempt, pod lineage), not by event order, so changing a
  placement does not shift any other request's random variates.
- A fixed `(config, seed, scorer, proposer)` gives a bitwise-identical `EpisodeResult.digest()`
  within the pinned environment (interpreter, dependencies, CPU architecture). Cross-platform
  bitwise identity is not claimed.
- Equal-time events run in scheduling order; where same-instant ordering matters (energy accrual
  before a power change, victim locking), it is enforced inside a single process step.
