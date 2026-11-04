"""Run-metadata tracking (TODO 0.5): SQLite metadata + parquet metrics + explicit seeds.

See ``docs/SEEDS.md`` for the seed-management convention and ``analysis/README.md`` for the
storage layout. Everything here is deterministic given its inputs: no wall-clock, no OS
entropy, no implicit defaults.
"""

from analysis.tracking.seeds import derive_seed, spawn_rng
from analysis.tracking.store import RunRecord, RunStore

__all__ = ["RunRecord", "RunStore", "derive_seed", "spawn_rng"]
