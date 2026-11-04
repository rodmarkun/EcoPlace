# analysis/ — run tracking and seeds

- `tracking/store.py` — run-metadata store: SQLite (`tracking/schema.sql`) for run id, component,
  experiment, root seed, config hash and JSON, git commit, status and timestamps; one parquet file
  of long-format metrics per run (`runs/<run_id>/metrics.parquet`).
- `tracking/seeds.py` — deterministic seed derivation from a run's root seed (see
  [`docs/SEEDS.md`](../docs/SEEDS.md)).
- `tracking/example.py` — minimal working example; tests in `tests/test_tracking.py`.

The statistical analyses of each study live with the study, in `experiments/<study>/analyze.py`.
