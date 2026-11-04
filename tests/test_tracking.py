"""Tests for analysis/tracking (TODO 0.5): seed derivation + run store."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest

from analysis.tracking import RunStore, derive_seed, spawn_rng

T0 = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)
T1 = datetime(2026, 7, 13, 12, 5, 0, tzinfo=UTC)


# --- seeds ----------------------------------------------------------------


def test_derive_seed_is_deterministic_and_in_range():
    a = derive_seed(42, "sim", "topology", 3)
    b = derive_seed(42, "sim", "topology", 3)
    assert a == b
    assert 0 <= a < 2**63


def test_derive_seed_distinguishes_paths_and_roots():
    seeds = {
        derive_seed(42, "sim", "topology", 3),
        derive_seed(42, "sim", "topology", 4),
        derive_seed(42, "sim", "workload", 3),
        derive_seed(43, "sim", "topology", 3),
        # encoding must not collapse adjacent elements or int/str
        derive_seed(42, "sim", "topology3"),
        derive_seed(42, "sim", "topology", "3"),
    }
    assert len(seeds) == 6


def test_derive_seed_rejects_missing_or_bad_inputs():
    with pytest.raises(ValueError):
        derive_seed(42)  # empty path: caller forgot to identify the component
    with pytest.raises(ValueError):
        derive_seed(-1, "x")
    with pytest.raises(TypeError):
        derive_seed("42", "x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        derive_seed(42, 3.14)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        derive_seed(True, "x")  # bools are not seeds


def test_derive_seed_has_no_default_seed():
    # The convention's teeth: you cannot call these without an explicit seed.
    with pytest.raises(TypeError):
        derive_seed()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        spawn_rng()  # type: ignore[call-arg]


def test_spawn_rng_reproducible_streams():
    x = spawn_rng(7, "gp", "init").random(4)
    y = spawn_rng(7, "gp", "init").random(4)
    z = spawn_rng(7, "gp", "mutate").random(4)
    assert (x == y).all()
    assert not (x == z).all()


# --- store ----------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    return RunStore(tmp_path / "trackdb")


def _register(store: RunStore, run_id: str = "r-001", **kw):
    defaults = dict(
        component="tests.tracking",
        experiment_id="E0",
        root_seed=123,
        config={"b": 2, "a": 1},
        created_utc=T0,
        git_commit="deadbeef",
    )
    defaults.update(kw)
    return store.register_run(run_id, **defaults)


def test_register_and_get_roundtrip(store):
    run = _register(store)
    got = store.get_run("r-001")
    assert got == run
    assert got.status == "created"
    assert got.config == {"a": 1, "b": 2}
    assert got.root_seed == 123
    assert got.created_utc == T0.isoformat()


def test_config_hash_is_key_order_independent(store):
    a = _register(store, "r-a", config={"a": 1, "b": 2})
    b = _register(store, "r-b", config={"b": 2, "a": 1})
    assert a.config_sha256 == b.config_sha256


def test_duplicate_run_id_rejected(store):
    _register(store)
    with pytest.raises(sqlite3.IntegrityError):
        _register(store)


def test_naive_datetime_rejected(store):
    with pytest.raises(ValueError, match="tz-aware"):
        _register(store, created_utc=datetime(2026, 7, 13, 12, 0, 0))  # noqa: DTZ001


def test_status_lifecycle_and_unknown_run(store):
    _register(store)
    store.set_status("r-001", "running", updated_utc=T1)
    assert store.get_run("r-001").status == "running"
    assert store.get_run("r-001").updated_utc == T1.isoformat()
    with pytest.raises(ValueError):
        store.set_status("r-001", "exploded", updated_utc=T1)
    with pytest.raises(KeyError):
        store.set_status("nope", "running", updated_utc=T1)


def test_metrics_roundtrip_and_convention_path(store):
    _register(store)
    rows = [
        {"step": 0, "metric": "energy_wh", "value": 10.5},
        {"step": 1, "metric": "energy_wh", "value": 11.0},
        {"step": 1, "metric": "p95_ms", "value": 240.0},
    ]
    path = store.write_metrics("r-001", rows)
    assert path == store.root / "runs" / "r-001" / "metrics.parquet"
    table = store.read_metrics("r-001")
    assert table.num_rows == 3
    assert set(table.column_names) == {"run_id", "step", "metric", "value"}
    assert table.column("run_id").to_pylist() == ["r-001"] * 3
    assert store.get_run("r-001").metrics_path == "runs/r-001/metrics.parquet"


def test_metrics_require_registered_run_and_rows(store):
    with pytest.raises(KeyError):
        store.write_metrics("ghost", [{"step": 0, "metric": "m", "value": 1.0}])
    _register(store)
    with pytest.raises(ValueError):
        store.write_metrics("r-001", [])
    with pytest.raises(FileNotFoundError):
        store.read_metrics("r-001")


def test_list_runs_filters_by_experiment(store):
    _register(store, "r-1", experiment_id="E1")
    _register(store, "r-2", experiment_id="E3")
    _register(store, "r-3", experiment_id="E1")
    assert store.list_runs() == ["r-1", "r-2", "r-3"]
    assert store.list_runs(experiment_id="E1") == ["r-1", "r-3"]


def test_register_run_enforces_root_seed_domain(store):
    """Phase 0 review: the documented [0, 2**63) bound (docs/SEEDS.md) is enforced by
    the store, matching seeds.derive_seed."""
    for bad in (-1, 2**63, True):
        with pytest.raises(ValueError):
            _register(store, "r-bad", root_seed=bad)
    _register(store, "r-max", root_seed=2**63 - 1)  # top of the domain is legal
