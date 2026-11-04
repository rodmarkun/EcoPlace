"""SQLite + parquet run-metadata store (TODO 0.5).

Layout under a caller-chosen ``store_root``::

    <store_root>/runs.sqlite                     run metadata (schema.sql)
    <store_root>/runs/<run_id>/metrics.parquet   bulk metrics, long/tidy format

Design choices:

- **Timestamps are injected**, never defaulted: the library takes tz-aware datetimes so that
  replayed/simulated runs can carry logical time and tests are deterministic.
- **Metrics are long-format** (``run_id, step, metric, value``): schema-stable across phases,
  trivially concatenable across runs for analysis.
- Heavy artifacts (evolution archives, traces) are out of scope here; they get git-lfs/DVC when
  Phase 5 produces them. This store only records *where* they are (via config/metrics).
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from analysis.tracking.seeds import MAX_ROOT_SEED

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

METRICS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("step", pa.int64(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=False),
    ]
)


def _canonical_json(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_utc(name: str, value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a tz-aware datetime (docs/SEEDS.md: no naive time)")
    return value.isoformat()


@dataclass(frozen=True)
class RunRecord:
    """One row of the ``runs`` table."""

    run_id: str
    component: str
    experiment_id: str | None
    root_seed: int
    config: dict[str, Any]
    config_sha256: str
    git_commit: str | None
    status: str
    created_utc: str
    updated_utc: str
    metrics_path: str | None


class RunStore:
    """Run-metadata store over one SQLite file + a parquet directory tree."""

    def __init__(self, store_root: Path | str):
        self.root = Path(store_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._db_path = self.root / "runs.sqlite"
        with self._connect() as conn:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- metadata ---------------------------------------------------------

    def register_run(
        self,
        run_id: str,
        *,
        component: str,
        root_seed: int,
        config: dict[str, Any],
        created_utc: datetime,
        experiment_id: str | None = None,
        git_commit: str | None = None,
    ) -> RunRecord:
        """Insert a new run in status ``created``. ``run_id`` must be unique."""
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"run_id must match {_RUN_ID_RE.pattern!r}, got {run_id!r}")
        if (
            isinstance(root_seed, bool)
            or not isinstance(root_seed, int)
            or not (0 <= root_seed <= MAX_ROOT_SEED)
        ):
            raise ValueError(
                f"root_seed must be an int in [0, 2**63), got {root_seed!r} (docs/SEEDS.md)"
            )
        created = _require_utc("created_utc", created_utc)
        config_json = _canonical_json(config)
        config_sha = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, component, experiment_id, root_seed, config_json,
                                     config_sha256, git_commit, status, created_utc, updated_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'created', ?, ?)""",
                (
                    run_id,
                    component,
                    experiment_id,
                    root_seed,
                    config_json,
                    config_sha,
                    git_commit,
                    created,
                    created,
                ),
            )
        return self.get_run(run_id)

    def set_status(self, run_id: str, status: str, *, updated_utc: datetime) -> None:
        """Move a run to ``running``/``completed``/``failed``."""
        if status not in ("created", "running", "completed", "failed"):
            raise ValueError(f"invalid status {status!r}")
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, updated_utc = ? WHERE run_id = ?",
                (status, _require_utc("updated_utc", updated_utc), run_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"unknown run_id {run_id!r}")

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch one run's metadata."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        return RunRecord(
            run_id=row["run_id"],
            component=row["component"],
            experiment_id=row["experiment_id"],
            root_seed=row["root_seed"],
            config=json.loads(row["config_json"]),
            config_sha256=row["config_sha256"],
            git_commit=row["git_commit"],
            status=row["status"],
            created_utc=row["created_utc"],
            updated_utc=row["updated_utc"],
            metrics_path=row["metrics_path"],
        )

    def list_runs(self, *, experiment_id: str | None = None) -> list[str]:
        """Run ids, optionally filtered by experiment, in insertion order."""
        query = "SELECT run_id FROM runs"
        params: tuple[str, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY rowid"
        with self._connect() as conn:
            return [r["run_id"] for r in conn.execute(query, params)]

    # -- metrics ----------------------------------------------------------

    def metrics_path(self, run_id: str) -> Path:
        """Parquet convention: ``<store_root>/runs/<run_id>/metrics.parquet``."""
        return self.root / "runs" / run_id / "metrics.parquet"

    def write_metrics(self, run_id: str, rows: list[dict[str, Any]]) -> Path:
        """Write long-format metric rows (``step``, ``metric``, ``value``) for a run.

        Overwrites any previous metrics file for the run (runs are immutable units; partial
        re-writes are a smell — re-run instead).
        """
        self.get_run(run_id)  # raises KeyError for unknown runs
        if not rows:
            raise ValueError("rows must be non-empty")
        table = pa.Table.from_pylist(
            [
                {
                    "run_id": run_id,
                    "step": int(r["step"]),
                    "metric": str(r["metric"]),
                    "value": float(r["value"]),
                }
                for r in rows
            ],
            schema=METRICS_SCHEMA,
        )
        path = self.metrics_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET metrics_path = ? WHERE run_id = ?",
                (str(path.relative_to(self.root)), run_id),
            )
        return path

    def read_metrics(self, run_id: str) -> pa.Table:
        """Read a run's metrics back as a pyarrow Table."""
        path = self.metrics_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"no metrics written for run {run_id!r} ({path})")
        return pq.read_table(path)
