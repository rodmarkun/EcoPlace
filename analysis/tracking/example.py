"""Minimal working example of the tracking store + seed convention (TODO 0.5).

Run it:  uv run python analysis/tracking/example.py
It writes into a temporary directory and prints what it did — no repo state is touched.
"""

from __future__ import annotations

import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from analysis.tracking import RunStore, derive_seed, spawn_rng


def main() -> None:
    root_seed = 20260713  # the run's single explicit root seed (docs/SEEDS.md)

    with tempfile.TemporaryDirectory(prefix="evolved-continuum-example-") as tmp:
        store = RunStore(Path(tmp))

        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None

        run = store.register_run(
            "example-0001",
            component="docs.example",
            experiment_id=None,
            root_seed=root_seed,
            config={"purpose": "0.5 minimal example", "episodes": 3},
            created_utc=datetime.now(tz=UTC),  # injected at the call site, not in the library
            git_commit=git_commit,
        )
        print(f"registered {run.run_id} (config sha256 {run.config_sha256[:12]}…)")

        store.set_status(run.run_id, "running", updated_utc=datetime.now(tz=UTC))

        # Every stochastic component derives its own seed from the root seed + its path.
        rows = []
        for episode in range(3):
            rng = spawn_rng(root_seed, "example", "episode", episode)
            energy_wh = float(10 + 2 * rng.random())  # stand-in for a measured quantity
            rows.append({"step": episode, "metric": "energy_wh", "value": energy_wh})
            print(
                f"episode {episode}: seed={derive_seed(root_seed, 'example', 'episode', episode)}"
                f" energy={energy_wh:.3f} Wh"
            )

        path = store.write_metrics(run.run_id, rows)
        store.set_status(run.run_id, "completed", updated_utc=datetime.now(tz=UTC))

        table = store.read_metrics(run.run_id)
        print(f"wrote {table.num_rows} metric rows to {path}")
        print(f"final status: {store.get_run(run.run_id).status}")


if __name__ == "__main__":
    main()
