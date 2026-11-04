-- Run-metadata store (TODO 0.5). One row per run; bulk metrics live in parquet at
-- <store_root>/runs/<run_id>/metrics.parquet (see analysis/README.md).
-- Applied idempotently by RunStore.

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,             -- caller-chosen, unique, filesystem-safe
    component     TEXT NOT NULL,                -- e.g. 'evolution.campaign', 'sim.calibration'
    experiment_id TEXT,                         -- e.g. 'E1', 'E3', NULL for infra runs
    root_seed     INTEGER NOT NULL,             -- docs/SEEDS.md: the run's single root seed
    config_json   TEXT NOT NULL,                -- full config, canonical JSON
    config_sha256 TEXT NOT NULL,                -- hash of config_json (cheap equality checks)
    git_commit    TEXT,                         -- repo commit the run executed from
    status        TEXT NOT NULL
                  CHECK (status IN ('created', 'running', 'completed', 'failed')),
    created_utc   TEXT NOT NULL,                -- ISO-8601 UTC, injected by caller
    updated_utc   TEXT NOT NULL,                -- ISO-8601 UTC, injected by caller
    metrics_path  TEXT                          -- relative parquet path, set on first write
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs (experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_component ON runs (component);
