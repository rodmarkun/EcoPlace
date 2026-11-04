"""One-shot P1-registered DES verification run on acceptance fixtures (T6), v2.

v2 (24 September 2026) replaces the v1 per-seed M/M/1 tolerance, which a correct
simulator fails with probability ~0.38 at five seeds, with a pooled bias test over a
fresh 20-seed matrix plus a per-seed sanity bound; see
experiments/scientific_gates_20260920/DES_V2_DESIGN.md. The v1 attempt is retained.

The driver has no caller-selected criteria path. It only trusts the fixed
``prereg/P1_DES_GATE.json`` registration, which must pin the criteria, this driver,
the fixture manifest, the repository commit, the executed source closure, and the
execution environment. Before the first simulation it atomically creates a local
attempt marker. This is a procedural one-shot control under an explicitly registered
honest-operator assumption; a writable directory is not an append-only trust boundary.

Usage in the registered gate window only::

    CONFIRM_P1_GATE=1 uv run python sim/fixtures/acceptance/run_gate.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent
REPO = HERE.parents[2]
REGISTRATION = REPO / "prereg" / "P1_DES_GATE.json"
ATTEMPT = HERE / "GATE_ATTEMPT.json"
RESULT = HERE / "GATE_RESULT.json"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _atomic_create(path: Path, value: Any) -> None:
    """Create a local one-shot marker without overwriting an existing path.

    O_EXCL prevents accidental replacement and mode 0444 discourages edits. The owner
    of the writable parent directory can still unlink it, so preregistration treats
    this as an honest-operator procedural control, not irreversible evidence.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o444)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Retain a partial marker after a write/fsync failure so an honest operator
        # cannot accidentally retry without first noticing the failed attempt.
        raise


def fail(msg: str) -> None:
    print(f"GATE REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "uv_lock_sha256": _sha_bytes((REPO / "uv.lock").read_bytes()),
    }


def _dirty_source_entries() -> tuple[str, ...]:
    """Return dirty/untracked inputs that can change this gate's executed code."""
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "sim",
            "analysis/tracking",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return _parse_dirty_source_status(status)


def _parse_dirty_source_status(status: str) -> tuple[str, ...]:
    """Pure parser kept separate so the pre-execution guard has a unit regression."""
    return tuple(line for line in status.splitlines() if line.strip())


def _execution_source_hashes() -> dict[str, str]:
    """Hash the Python closure executed by the gate plus the loaded native module.

    The dependency graph is intentionally small and explicit: the simulator package
    and its RNG helper are local Python; third-party dependency versions are frozen by
    ``uv.lock`` in the environment record. PlacementScript may be a compiled extension,
    so its actual loaded artifact is hashed rather than merely its Rust source tree.
    """
    paths = [
        *sorted((REPO / "sim").rglob("*.py")),
        *sorted((REPO / "analysis" / "tracking").rglob("*.py")),
    ]
    hashes = {path.relative_to(REPO).as_posix(): _sha_bytes(path.read_bytes()) for path in paths}
    shim_path, native_path = _placementscript_artifacts()
    hashes["@python/placementscript/__init__.py"] = _sha_bytes(shim_path.read_bytes())
    hashes[f"@native/{native_path.name}"] = _sha_bytes(native_path.read_bytes())
    return hashes


def _native_artifact_path() -> Path:
    """Compatibility helper for tests/reporting: resolve the compiled extension."""
    return _placementscript_artifacts()[1]


def _placementscript_artifacts() -> tuple[Path, Path]:
    """Resolve both executable package files without importing unverified code."""
    import importlib.machinery
    import importlib.util

    package = importlib.util.find_spec("placementscript")
    if package is None or package.origin is None or package.submodule_search_locations is None:
        fail("cannot locate the PlacementScript package")
    shim_path = Path(package.origin).resolve()
    package_root = Path(next(iter(package.submodule_search_locations))).resolve()
    native_matches = sorted(
        {
            path.resolve()
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for path in package_root.glob(f"placementscript*{suffix}")
            if path.is_file()
        }
    )
    if not shim_path.is_file() or len(native_matches) != 1:
        fail("expected one Python shim and one native PlacementScript extension")
    return shim_path, native_matches[0]


def _preflight() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str], list[int]
]:
    if os.environ.get("CONFIRM_P1_GATE") != "1":
        fail("CONFIRM_P1_GATE=1 not set — this is the one-shot P1 gate run")
    if ATTEMPT.exists() or RESULT.exists():
        fail("a gate attempt/result already exists; a new registration version is required")
    if not REGISTRATION.is_file():
        fail(f"fixed P1 registration is missing: {REGISTRATION}")
    registration = json.loads(REGISTRATION.read_text())
    required = {
        "status",
        "gate",
        "criteria_path",
        "criteria_sha256",
        "driver_sha256",
        "manifest_sha256",
        "environment_sha256",
        "one_shot_assumption",
        "source_commit",
        "source_closure_sha256",
    }
    if set(registration) != required:
        fail(f"P1 registration fields must be exactly {sorted(required)}")
    if registration["status"] != "frozen" or registration["gate"] != "phase2-des-v2":
        fail("P1 registration is not the frozen phase2-des-v2 registration")
    if registration["one_shot_assumption"] != "honest-operator-local-marker":
        fail("P1 registration does not state the required local-marker assumption")

    dirty = _dirty_source_entries()
    if dirty:
        fail("executed source tree is dirty or untracked: " + "; ".join(dirty))

    driver_sha = _sha_bytes(Path(__file__).read_bytes())
    manifest_path = FIXTURES / "MANIFEST.json"
    manifest_sha = _sha_bytes(manifest_path.read_bytes())
    environment = _environment()
    source_closure_sha = _canonical_sha(_execution_source_hashes())
    checks = {
        "driver_sha256": driver_sha,
        "manifest_sha256": manifest_sha,
        "environment_sha256": _canonical_sha(environment),
        "source_closure_sha256": source_closure_sha,
    }
    for field, actual in checks.items():
        if registration[field] != actual:
            fail(f"{field} does not match the frozen P1 registration")

    criteria_path = (REPO / registration["criteria_path"]).resolve()
    prereg_root = (REPO / "prereg").resolve()
    if prereg_root not in criteria_path.parents or not criteria_path.is_file():
        fail("registered criteria_path must be an existing file under prereg/")
    if _sha_bytes(criteria_path.read_bytes()) != registration["criteria_sha256"]:
        fail("criteria file does not match the frozen P1 hash")
    criteria = json.loads(criteria_path.read_text())

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != registration["source_commit"]:
        fail("repository HEAD does not match the source commit frozen at P1")

    manifest = json.loads(manifest_path.read_text())
    for rel, expected in manifest["files"].items():
        path = FIXTURES / rel
        if _sha_bytes(path.read_bytes()) != expected:
            fail(f"{rel} hash mismatch vs MANIFEST.json — not the frozen artifact")
    seeds = manifest["acceptance_seed_matrix"]
    return registration, criteria, manifest, environment, seeds


def _execute(
    criteria: dict[str, Any], manifest: dict[str, Any], seeds: list[int]
) -> dict[str, Any]:
    from sim import CallableScorer, load_episode_config, run_episode
    from sim.fixtures.make_fixtures import ACCEPTANCE_SEEDS

    if list(ACCEPTANCE_SEEDS) != seeds:
        raise RuntimeError("seed matrix drifted between generator and manifest")
    flat = CallableScorer("flat", lambda pod, node, ctx: 1.0)
    outcomes: dict[str, dict[str, Any]] = {}

    c = criteria["mm1"]
    config = load_episode_config(HERE / "mm1.json")
    w_closed = 1.0 / (c["mu"] - c["rate"])
    errors, total, count = [], 0.0, 0
    for seed in seeds:
        result = run_episode(config, seed=seed, scorer=flat)
        stats = result.latency_stats(c["warmup_s"], config.horizon_s - c["margin_s"])
        if stats["count"] < criteria["min_samples"]:
            raise RuntimeError(f"mm1 seed {seed}: too few samples")
        errors.append((stats["mean"] - w_closed) / w_closed)
        total += stats["mean"] * stats["count"]
        count += int(stats["count"])
    pooled_error = (total / count - w_closed) / w_closed
    outcomes["mm1"] = {
        "closed_form_w": w_closed,
        "signed_relative_errors": errors,
        "pooled_relative_error": pooled_error,
        "pooled_tolerance": c["pooled_tolerance"],
        "per_seed_bound": c["per_seed_bound"],
        "pass": abs(pooled_error) < c["pooled_tolerance"]
        and all(abs(e) < c["per_seed_bound"] for e in errors),
    }

    c = criteria["jackson"]
    config = load_episode_config(HERE / "jackson.json")
    t_closed = 1.0 / (c["mu_a"] - c["rate"]) + c["p"] * (1.0 / (c["mu_b"] - c["p"] * c["rate"]))
    total, count = 0.0, 0
    for seed in seeds:
        result = run_episode(config, seed=seed, scorer=flat)
        stats = result.latency_stats(
            criteria["mm1"]["warmup_s"],
            config.horizon_s - criteria["mm1"]["margin_s"],
        )
        total += stats["mean"] * stats["count"]
        count += int(stats["count"])
    pooled = total / count
    relative_error = abs(pooled - t_closed) / t_closed
    outcomes["jackson"] = {
        "closed_form_t": t_closed,
        "pooled_mean": pooled,
        "relative_error": relative_error,
        "tolerance": c["tolerance"],
        "pass": relative_error < c["tolerance"],
    }

    config = load_episode_config(HERE / "overload.json")
    drops = [
        run_episode(config, seed=seed, scorer=flat).ledger["requests_dropped_queue_full"]
        for seed in seeds
    ]
    outcomes["overload"] = {"drops_per_seed": drops, "pass": all(d > 0 for d in drops)}

    config = load_episode_config(HERE / "small_cluster.json")
    digests = [run_episode(config, seed=seeds[0], scorer=flat).digest() for _ in range(2)]
    outcomes["replay"] = {"digests": digests, "pass": digests[0] == digests[1]}
    return {
        "manifest_generator_sha256": manifest["generator_sha256"],
        "seed_matrix": seeds,
        "outcomes": outcomes,
        "gate_pass": all(outcome["pass"] for outcome in outcomes.values()),
    }


def main() -> None:
    registration, criteria, manifest, environment, seeds = _preflight()
    started = datetime.datetime.now(datetime.UTC).isoformat()
    attempt = {
        "status": "started",
        "started_at_utc": started,
        "registration_sha256": _sha_bytes(REGISTRATION.read_bytes()),
        "source_commit": registration["source_commit"],
        "source_closure_sha256": registration["source_closure_sha256"],
        "one_shot_assumption": registration["one_shot_assumption"],
        "environment": environment,
    }
    # Procedural boundary: no acceptance simulation may occur before this succeeds.
    _atomic_create(ATTEMPT, attempt)
    try:
        record = _execute(criteria, manifest, seeds)
    except BaseException as exc:
        failure = {
            "status": "failed",
            "started_at_utc": started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "source_commit": registration["source_commit"],
            "source_closure_sha256": registration["source_closure_sha256"],
            "one_shot_assumption": registration["one_shot_assumption"],
        }
        if not RESULT.exists():
            _atomic_create(RESULT, failure)
        raise
    record.update(
        {
            "status": "passed" if record["gate_pass"] else "failed",
            "started_at_utc": started,
            "finished_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "source_commit": registration["source_commit"],
            "source_closure_sha256": registration["source_closure_sha256"],
            "one_shot_assumption": registration["one_shot_assumption"],
        }
    )
    _atomic_create(RESULT, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(0 if record["gate_pass"] else 1)


if __name__ == "__main__":
    main()
