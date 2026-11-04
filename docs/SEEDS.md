# Seed-management convention

**Rule: every stochastic component takes an explicit seed. There are no wall-clock or
OS-entropy defaults anywhere in experiment-relevant code.** Common random numbers and exact
replay depend on it.

## The scheme

1. **One root seed per run.** Every run registered in the tracking store
   (`analysis/tracking`) records a single integer `root_seed` in `[0, 2**63)`. Seed matrices
   for campaigns/episodes are defined in experiment configs as lists of root seeds.
2. **Deterministic derivation, never sharing.** A component never uses the root seed
   directly and never shares an RNG with another component. It derives its own seed:

   ```python
   from analysis.tracking import derive_seed, spawn_rng

   seed = derive_seed(root_seed, "sim", "topology-gen", family, instance_idx)
   rng  = spawn_rng(root_seed, "evolution", "island", island_idx)   # numpy Generator
   ```

   Derivation is SHA-256 over a length-prefixed, type-tagged encoding of
   `(root_seed, *path)` — stable across processes, platforms, and Python versions, and free
   of separator collisions. The *path* is the component's identity; adding a parallel worker
   or reordering initialization never perturbs anyone else's stream.
3. **Cross-language consumers** (Rust sim components, GP baselines, PPO) receive the derived
   integer seed explicitly at their boundary (function argument, config field, CLI flag).
   Derived seeds fit in `u64`/`i64` by construction (63 bits).
4. **Common random numbers** (C13): paired method comparisons reuse the *same root seed and
   the same environment paths* across methods, so environment randomness (workload, churn,
   topology) is identical and only the policy differs. Method-internal randomness uses
   method-specific paths (e.g. `("evolution", ...)` vs `("gp", ...)`).

## Banned constructions

- `np.random.default_rng()` / `np.random.seed()` with no argument, and the global
  `np.random.*` functions in experiment code.
- `random.Random()` / `random.seed()` with no argument; the module-level `random.*` functions.
- `torch.manual_seed` *omitted* (PPO baseline must set it from a derived seed; also set
  deterministic flags where supported).
- `time.time()`, `datetime.now()`, `uuid4()`, `os.urandom()` as inputs to anything that
  influences results. (Naive `datetime.now()` is additionally flagged by ruff's `DTZ` rules;
  the tracking library refuses naive datetimes outright.)
- Default-valued seed parameters (`def f(..., seed=0)` or `seed=None`). Seeds are required,
  positional-or-keyword, with **no default** — omission must be a `TypeError`, not a quiet
  fixed stream (a fixed default silently correlates supposedly independent replications).

## Enforcement

- `analysis/tracking/seeds.py` raises on missing/invalid seeds and paths.
- Review checklist item in `CONTRIBUTING.md` (rule 2).
- `tests/test_sim_invariants.py` applies a grep-based guard for every banned
  construction across the experiment-relevant Python surfaces under `sim/`,
  `baselines/`, and `evolution/`. The only exclusions are named operational/build-time
  tools whose timezone-aware timestamps never feed scientific results.

## Non-goals

Operational timestamps (run bookkeeping, logs) may use real wall-clock time — they are
*recorded*, tz-aware, and never feed back into results. The tracking store takes injected
timestamps so tests stay deterministic.
