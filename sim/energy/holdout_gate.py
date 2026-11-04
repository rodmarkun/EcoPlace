"""Run-once held-out EnergyModelBundle evaluation driver (TODO 3.3; plan §4.1, T4).

The held-out structural bundle set is evaluated **once**, after algorithm code and
tuning are frozen at a tagged commit, against the frozen champion policies named in
the P2 registration. This driver is the only sanctioned access path; it mirrors the
honest-operator pattern of the P1 DES gate driver ``run_gate.py`` (see
``prereg/README.md``):

1. **Registration first:** refuses to run unless ``prereg/P2_HELD_OUT_BUNDLES.json``
   exists and pins the bundle list hash (full canonical manifests — curve points,
   capacities, transitions, fitted exponents, swept ranges), the evaluation
   entrypoint **and its file hash**, and this driver's own hash. (P2 is frozen
   Sep 8; the registration file is written then — this module never creates it.)
2. **Tagged commit + full dependency closure (R2-4):** refuses unless HEAD is
   exactly a git tag, **the entire repository working tree matches the pinned
   commit** (``git status --porcelain --untracked-files=all`` over the whole repo
   must be empty — a dirty ``evolution/`` scenario or baseline module refuses the
   gate exactly like a dirty ``sim/`` file), the registration's pinned commit
   matches HEAD, **and the on-disk registration is byte-identical to the
   registration blob committed at HEAD** — a dirty or swapped registration cannot
   redirect execution.
3. **Pinned entrypoint, resolved without imports (R2-4):** the registered
   ``module:function`` is resolved to a repository file **textually**
   (``pkg/mod.py`` / ``pkg/mod/__init__.py``) — never via ``importlib`` machinery,
   which would execute parent packages from the working tree before any check —
   and must be tracked at HEAD, byte-identical to its HEAD blob, and matching the
   registered ``entrypoint_sha256``.
4. **Run-once marker:** atomically creates ``HOLDOUT_ATTEMPT.json`` (``O_EXCL``,
   mode 0444) *before* any evaluation, so a second casual invocation fails loudly.
5. **Explicit confirmation:** requires ``CONFIRM_HELD_OUT_GATE=1`` in the
   environment.
6. **Clean detached execution (R2-4):** the evaluation itself runs in a subprocess
   from a temporary **detached ``git worktree`` at the pinned commit**. Every
   out-of-repository editable is independently materialized as a detached worktree
   at its registered commit (never imported from the live sibling checkout), then
   ``uv sync --frozen`` reconstructs the environment from the detached tree's
   registered lockfile using exactly the dependency groups/extras frozen by P2. A
   locked PlacementScript wheel is built from that tree's Rust sources, installed
   into the fresh environment, and genuinely imported. The actual
   interpreter/platform, uv, Maturin, Rust/Cargo versions, lock hashes, wheel hash,
   installed native-module hashes, and a canonical names+versions hash of the
   installed distribution set are attested and checked against the P2 execution
   pins. Registered dependency modules and the entrypoint itself are imported from
   that environment **before** the run-once marker is created (R5-4/R4-5/R4-6).

**Honest enforcement claim** (do not overstate it): every check above is a
*procedural* control against accidents under the registered
``one_shot_assumption: honest-operator-local-marker``. A dishonest operator can
delete the marker, monkeypatch the module, or import the bundles directly with
``acknowledged_one_shot=True`` — no writable working tree can prevent that. The
durable provenance boundary is the timestamped, committed P2 registration (and the
public record of the tagged commit); anything beyond that would need external
append-only storage, which the registration's ``one_shot_assumption`` field states
explicitly.

The pre-execution checks are pure functions over an injected
:class:`GateEnvironment`, so tests exercise every refusal path without touching the
real repository state or the real marker.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTRATION = REPO / "prereg" / "P2_HELD_OUT_BUNDLES.json"
ATTEMPT = HERE / "HOLDOUT_ATTEMPT.json"
RESULT = HERE / "HOLDOUT_RESULT.json"

_EXECUTION_PIN_KEYS = frozenset(
    {
        "python_version",
        "python_implementation",
        "python_platform",
        "machine",
        "system",
        "uv_version",
        "maturin_version",
        "rustc_version",
        "cargo_version",
        "installed_distributions_sha256",
    }
)

_DEPENDENCY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _is_hex_digest(value: object, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _registered_name_list(registration: Mapping[str, object], key: str) -> tuple[str, ...]:
    """Validate a canonical P2 list of dependency selectors/import modules."""
    value = registration.get(key)
    if not isinstance(value, list):
        raise GateRefused(f"{key} must be a JSON list")
    if any(
        not isinstance(name, str) or _DEPENDENCY_NAME.fullmatch(name) is None
        for name in value
    ):
        raise GateRefused(
            f"{key} entries must be non-empty dependency/module names containing "
            "only letters, digits, '.', '_' or '-'"
        )
    if value != sorted(set(value)):
        raise GateRefused(f"{key} must be sorted and contain no duplicates")
    return tuple(value)


class GateRefused(RuntimeError):
    """Raised (and mapped to exit code 2) when any pre-execution check fails."""


@dataclass(frozen=True)
class EditableDepState:
    """State of one out-of-repository editable dependency (R3-3): the held-out
    evaluation's import closure includes these (e.g. the ShinkaEvolve fork), so the
    gate requires each to be a clean git checkout at the commit the P2 registration
    pins."""

    name: str
    path: str  # resolved absolute path
    commit: str | None  # None -> not a git checkout (refused)
    dirty: bool


@dataclass(frozen=True)
class GateEnvironment:
    """Everything the checks read, gathered once (or injected by tests)."""

    head_commit: str
    exact_tag: str | None  # `git describe --exact-match --tags HEAD`, None if untagged
    dirty_entries: tuple[str, ...]
    registration_exists: bool
    registration: dict | None
    #: raw bytes of the registration blob committed at HEAD (None if not committed)
    registration_head_bytes: bytes | None
    #: raw bytes of the on-disk registration file (None if absent)
    registration_disk_bytes: bytes | None
    attempt_marker_exists: bool
    confirm_env: str | None
    #: out-of-repository editable dependencies from uv.lock (R3-3)
    editable_deps: tuple[EditableDepState, ...] = ()


@dataclass(frozen=True)
class PreparedExecution:
    """Detached, frozen-synced execution tree prepared before the one-shot marker.

    ``attestation`` describes the *actual* worktree interpreter/toolchain and the
    built+installed PlacementScript wheel, rather than the gate process environment.
    """

    root: Path
    worktree: Path
    python: Path
    attestation: dict[str, object]


def editable_out_of_repo_deps(repo: Path) -> tuple[tuple[str, Path], ...]:
    """Parse ``uv.lock`` for editable path dependencies resolving *outside* the
    repository (R3-3): these are execution-relevant code the pinned commit cannot
    contain, so the registration must pin them separately."""
    import tomllib

    lock = repo / "uv.lock"
    if not lock.is_file():
        return ()
    data = tomllib.loads(lock.read_text())
    out: list[tuple[str, Path]] = []
    for pkg in data.get("package", []):
        editable = pkg.get("source", {}).get("editable")
        if editable is None:
            continue
        path = (repo / editable).resolve()
        if not path.is_relative_to(repo.resolve()):
            out.append((str(pkg["name"]), path))
    return tuple(out)


def gather_environment(repo: Path = REPO) -> GateEnvironment:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    registration_path = repo / "prereg" / "P2_HELD_OUT_BUNDLES.json"
    head = git("rev-parse", "HEAD")
    tag_proc = subprocess.run(
        ["git", "describe", "--exact-match", "--tags", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    exact_tag = tag_proc.stdout.strip() if tag_proc.returncode == 0 else None
    # R2-4 (dependency closure): the *whole repository* must match the pinned
    # commit — no path allowlist. A dirty evolution/ module, scenario file, or
    # baseline that the entrypoint imports refuses the gate like a dirty sim/ file.
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    repo_dirty_entries = tuple(line for line in status.splitlines() if line.strip())
    registration = None
    disk_bytes = None
    if registration_path.exists():
        disk_bytes = registration_path.read_bytes()
        registration = json.loads(disk_bytes)
    head_blob = subprocess.run(
        ["git", "show", f"HEAD:{registration_path.relative_to(repo).as_posix()}"],
        cwd=repo,
        capture_output=True,
    )
    head_bytes = head_blob.stdout if head_blob.returncode == 0 else None
    deps = []
    for name, path in editable_out_of_repo_deps(repo):
        rev = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        commit = rev.stdout.strip() if rev.returncode == 0 else None
        stat = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
            capture_output=True,
            text=True,
        )
        dep_dirty = bool(stat.stdout.strip()) if stat.returncode == 0 else True
        deps.append(EditableDepState(name=name, path=str(path), commit=commit, dirty=dep_dirty))
    return GateEnvironment(
        head_commit=head,
        exact_tag=exact_tag,
        dirty_entries=repo_dirty_entries,
        registration_exists=registration_path.exists(),
        registration=registration,
        registration_head_bytes=head_bytes,
        registration_disk_bytes=disk_bytes,
        attempt_marker_exists=ATTEMPT.exists(),
        confirm_env=os.environ.get("CONFIRM_HELD_OUT_GATE"),
        editable_deps=tuple(deps),
    )


def check_gate(env: GateEnvironment, *, driver_sha256: str) -> dict:
    """All refusal logic, pure. Returns the validated registration on success."""
    if env.attempt_marker_exists:
        raise GateRefused(
            f"attempt marker {ATTEMPT.name} already exists — the held-out evaluation is "
            "run-once; a repeat requires a new pre-registered protocol version, not a rerun"
        )
    if not env.registration_exists or env.registration is None:
        raise GateRefused(
            "prereg/P2_HELD_OUT_BUNDLES.json is missing — the held-out gate cannot run "
            "before P2 freezes the bundle list, entrypoint, and commit"
        )
    reg = env.registration
    required = (
        "pinned_commit",
        "driver_sha256",
        "bundle_list_sha256",
        "entrypoint",
        "entrypoint_sha256",
        "editable_dependencies",
        "dependency_groups",
        "dependency_extras",
        "entrypoint_dependency_imports",
        "uv_lock_sha256",
        "execution_environment",
    )
    for key in required:
        if key not in reg:
            raise GateRefused(f"registration is missing required key {key!r}")
    for key in ("driver_sha256", "bundle_list_sha256", "entrypoint_sha256", "uv_lock_sha256"):
        if not _is_hex_digest(reg[key], 64):
            raise GateRefused(f"{key} must be a 64-character hexadecimal digest")
    if not _is_hex_digest(reg["pinned_commit"], 40):
        raise GateRefused("pinned_commit must be a 40-character hexadecimal git object id")
    execution_pins = reg["execution_environment"]
    if not isinstance(execution_pins, dict) or set(execution_pins) != _EXECUTION_PIN_KEYS:
        raise GateRefused(
            "execution_environment must pin exactly " + ", ".join(sorted(_EXECUTION_PIN_KEYS))
        )
    if any(not isinstance(value, str) or not value for value in execution_pins.values()):
        raise GateRefused("every execution_environment pin must be a non-empty string")
    if not _is_hex_digest(execution_pins["installed_distributions_sha256"], 64):
        raise GateRefused(
            "execution_environment installed_distributions_sha256 must be a "
            "64-character hexadecimal digest"
        )
    _registered_name_list(reg, "dependency_groups")
    _registered_name_list(reg, "dependency_extras")
    dependency_imports = _registered_name_list(reg, "entrypoint_dependency_imports")
    if not dependency_imports:
        raise GateRefused(
            "entrypoint_dependency_imports must enumerate the registered "
            "entrypoint's import closure probes"
        )
    if any(
        not all(part.isidentifier() for part in module_name.split("."))
        for module_name in dependency_imports
    ):
        raise GateRefused(
            "entrypoint_dependency_imports entries must be dotted Python module names"
        )
    # R3-5: a malformed entrypoint must refuse HERE — before the run-once marker
    # burns — never crash after it.
    entry = str(reg["entrypoint"])
    if entry.count(":") != 1:
        raise GateRefused(f"registered entrypoint {entry!r} is not of the form module:function")
    module_name, func_name = entry.split(":")
    if (
        not module_name
        or not all(part.isidentifier() for part in module_name.split("."))
        or not func_name.isidentifier()
    ):
        raise GateRefused(
            f"registered entrypoint {entry!r} is not a dotted module path plus a "
            "function identifier"
        )
    # R3-3: every out-of-repository editable dependency (execution-relevant code
    # the pinned commit cannot contain) must be registered, clean, and at exactly
    # the registered commit.
    reg_deps = reg["editable_dependencies"]
    if not isinstance(reg_deps, dict):
        raise GateRefused("editable_dependencies must map dependency name -> pinned commit")
    if any(
        not isinstance(name, str) or not name or not _is_hex_digest(commit, 40)
        for name, commit in reg_deps.items()
    ):
        raise GateRefused(
            "editable_dependencies must map non-empty names to 40-character "
            "hexadecimal git object ids"
        )
    env_dep_names = {d.name for d in env.editable_deps}
    if set(reg_deps) != env_dep_names:
        raise GateRefused(
            "editable dependencies diverge from the registration: environment has "
            f"{sorted(env_dep_names)}, registration pins {sorted(reg_deps)} — an "
            "unregistered editable source cannot enter the held-out evaluation (R3-3)"
        )
    for dep in env.editable_deps:
        if dep.commit is None:
            raise GateRefused(
                f"editable dependency {dep.name} at {dep.path} is not a git "
                "checkout — its state cannot be attested (R3-3)"
            )
        if dep.dirty:
            raise GateRefused(
                f"editable dependency {dep.name} at {dep.path} has uncommitted "
                "changes — the held-out import closure must be fully pinned (R3-3)"
            )
        if dep.commit != str(reg_deps[dep.name]):
            raise GateRefused(
                f"editable dependency {dep.name} is at commit {dep.commit[:12]} but "
                f"the registration pins {str(reg_deps[dep.name])[:12]} (R3-3)"
            )
    if reg.get("one_shot_assumption") != "honest-operator-local-marker":
        raise GateRefused("registration must state the honest-operator one-shot assumption")
    if env.exact_tag is None:
        raise GateRefused(
            "HEAD is not at a tag — the held-out evaluation runs only at the frozen, "
            "tagged commit recorded in the P2 registration"
        )
    if env.head_commit != reg["pinned_commit"]:
        raise GateRefused(
            f"HEAD {env.head_commit[:12]} != registered commit {str(reg['pinned_commit'])[:12]}"
        )
    if env.registration_head_bytes is None:
        raise GateRefused(
            "the registration is not committed at HEAD — an uncommitted registration "
            "could redirect execution and is not a frozen protocol"
        )
    if env.registration_disk_bytes != env.registration_head_bytes:
        raise GateRefused(
            "the on-disk registration differs from the registration committed at "
            "HEAD — refusing to honor a modified registration"
        )
    if env.dirty_entries:
        raise GateRefused(
            "dirty/untracked files — the entire repository must match the pinned "
            "commit (R2-4 dependency closure): " + ", ".join(env.dirty_entries[:10])
        )
    if driver_sha256 != reg["driver_sha256"]:
        raise GateRefused("this driver's hash does not match the registered driver hash")
    if env.confirm_env != "1":
        raise GateRefused("set CONFIRM_HELD_OUT_GATE=1 to acknowledge the one-shot semantics")
    return reg


def resolve_module_path(module_name: str, root: Path) -> Path:
    """Resolve ``module_name`` to a file under ``root`` **textually** (R2-4): no
    ``importlib`` machinery, so no working-tree code — including parent packages —
    is ever executed during resolution. Accepts ``pkg/mod.py`` and
    ``pkg/mod/__init__.py`` layouts; refuses anything else."""
    if not module_name or not all(part.isidentifier() for part in module_name.split(".")):
        raise GateRefused(f"entrypoint module name {module_name!r} is not a dotted module path")
    rel = Path(*module_name.split("."))
    candidates = (root / rel.with_suffix(".py"), root / rel / "__init__.py")
    hits = [p for p in candidates if p.is_file()]
    if not hits:
        raise GateRefused(
            f"entrypoint module {module_name!r} does not resolve to a repository file "
            f"under {root} (looked for {[str(c) for c in candidates]})"
        )
    if len(hits) > 1:
        raise GateRefused(
            f"entrypoint module {module_name!r} is ambiguous: both "
            f"{hits[0]} and {hits[1]} exist — refusing to guess"
        )
    return hits[0].resolve()


def verify_entrypoint_pinned(module_name: str, registered_sha256: str) -> Path:
    """The registered entrypoint module must be a repo file, tracked at HEAD,
    byte-identical to its HEAD blob, and matching the registered hash (R1-8) —
    resolved without importing any working-tree code (R2-4)."""
    path = resolve_module_path(module_name, REPO)
    if not path.is_relative_to(REPO):
        raise GateRefused(f"entrypoint {path} lives outside the repository")
    rel = path.relative_to(REPO).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel], cwd=REPO, capture_output=True
    )
    if tracked.returncode != 0:
        raise GateRefused(f"entrypoint file {rel} is not tracked by git")
    head_blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO, capture_output=True)
    disk = path.read_bytes()
    if head_blob.returncode != 0 or head_blob.stdout != disk:
        raise GateRefused(f"entrypoint file {rel} differs from its blob at HEAD")
    if hashlib.sha256(disk).hexdigest() != registered_sha256:
        raise GateRefused(f"entrypoint file {rel} does not match the registered entrypoint_sha256")
    return path


def _atomic_create(path: Path, value: dict) -> None:
    """O_EXCL marker creation (accidental-rerun control, per the P1 gate pattern)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bundle_list_sha256() -> str:
    """Canonical hash of the held-out bundle definitions (pinned by P2): the **full
    manifests** — curve points, capacities, transition parameters, fitted exponents,
    swept ranges, coefficients, roles (R1-8: an id-level hash would not detect a
    numerically altered bundle)."""
    from sim.energy.bundles import build_held_out_bundles
    from sim.energy.sources import load_pinned_sources

    registry = load_pinned_sources()
    bundles = build_held_out_bundles(registry, acknowledged_one_shot=True)
    payload = [b.manifest() for b in bundles]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# Runner executed *inside the detached pinned worktree* (R2-4/R3-3): imports
# resolve against the pinned tree and its own frozen-synced environment, so the
# entrypoint and its whole import closure come from the registered commit + lock.
# json.dump refuses NaN/Infinity (R3-5): held-out results must be finite.
_WORKTREE_RUNNER = """\
import importlib
import json
import sys

worktree, module_name, func_name, out_path = sys.argv[1:5]
sys.path.insert(0, worktree)
import placementscript
from sim.energy.bundles import build_held_out_bundles
from sim.energy.sources import load_pinned_sources

assert placementscript.Policy is not None
func = getattr(importlib.import_module(module_name), func_name)
bundles = build_held_out_bundles(load_pinned_sources(), acknowledged_one_shot=True)
with open(out_path, "w") as fh:
    json.dump(func(bundles), fh, allow_nan=False)
"""

_RUNTIME_ATTESTATION_PROBE = r"""
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import platform
import re
import sysconfig
from pathlib import Path

import placementscript

if placementscript.Policy is None:
    raise RuntimeError("placementscript native API did not load")
spec = importlib.util.find_spec("placementscript")
if spec is None or spec.origin is None or spec.submodule_search_locations is None:
    raise RuntimeError("cannot resolve installed placementscript package")
shim = Path(spec.origin).resolve()
root = Path(next(iter(spec.submodule_search_locations))).resolve()
native = sorted({
    path.resolve()
    for suffix in importlib.machinery.EXTENSION_SUFFIXES
    for path in root.glob(f"placementscript*{suffix}")
    if path.is_file()
})
if not shim.is_file() or len(native) != 1:
    raise RuntimeError("expected one placementscript shim and one native extension")
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
installed = sorted({
    (
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(),
        dist.version,
    )
    for dist in importlib.metadata.distributions()
    if dist.metadata["Name"] and dist.version
})
installed_payload = [
    {"name": name, "version": version}
    for name, version in installed
]
installed_sha256 = hashlib.sha256(
    json.dumps(
        installed_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "python_platform": sysconfig.get_platform(),
    "machine": platform.machine(),
    "system": platform.system(),
    "placementscript_shim_sha256": sha(shim),
    "placementscript_native_sha256": sha(native[0]),
    "placementscript_native_name": native[0].name,
    "installed_distributions": installed_payload,
    "installed_distributions_sha256": installed_sha256,
}))
"""

_ENTRYPOINT_PREFLIGHT = r"""
import importlib
import json
import sys

worktree, module_name, func_name, dependency_json = sys.argv[1:5]
sys.path.insert(0, worktree)
for dependency in json.loads(dependency_json):
    importlib.import_module(dependency)
entrypoint = getattr(importlib.import_module(module_name), func_name)
if not callable(entrypoint):
    raise TypeError(f"registered entrypoint {module_name}:{func_name} is not callable")
"""


def result_json(payload: dict) -> str:
    """Canonical held-out result serialization (R3-5): NaN/Infinity anywhere in the
    result payload is a hard error, never a silently emitted non-standard token."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _checked_output(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _uv_sync_command(
    python_version: str,
    dependency_groups: tuple[str, ...],
    dependency_extras: tuple[str, ...],
) -> list[str]:
    """Build the exact frozen-sync command from the P2 dependency selection."""
    command = [
        "uv",
        "sync",
        "--frozen",
        "--no-default-groups",
        "--python",
        python_version,
    ]
    for group in dependency_groups:
        command.extend(("--group", group))
    for extra in dependency_extras:
        command.extend(("--extra", extra))
    return command


def _installed_distributions_sha256(value: object) -> str:
    """Re-hash the attested canonical ``[{name, version}, ...]`` distribution set."""
    if not isinstance(value, list) or not value:
        raise GateRefused("installed_distributions attestation must be a non-empty list")
    pairs: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise GateRefused(
                "each installed distribution attestation must contain exactly name and version"
            )
        name, version = item["name"], item["version"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
        ):
            raise GateRefused("installed distribution names and versions must be non-empty strings")
        pairs.append({"name": name, "version": version})
    canonical = sorted(pairs, key=lambda item: (item["name"], item["version"]))
    if pairs != canonical or len({item["name"] for item in pairs}) != len(pairs):
        raise GateRefused(
            "installed distribution attestation must be sorted with unique canonical names"
        )
    return hashlib.sha256(
        json.dumps(
            pairs, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _build_and_install_native_runtime(
    worktree: Path,
    *,
    builder_python: Path,
    install_python: Path,
    build_root: Path,
    env: dict[str, str],
) -> str:
    """Build a wheel from the pinned Rust sources and install that exact wheel.

    This runs before the one-shot marker. ``--locked`` binds Cargo resolution to the
    worktree's ``runtime/Cargo.lock``; the returned wheel hash is written into the
    attempt attestation (R4-5).
    """
    wheel_dir = build_root / "placementscript-wheel"
    wheel_dir.mkdir(parents=True, exist_ok=False)
    manifest = worktree / "runtime" / "crates" / "placementscript-py" / "Cargo.toml"
    subprocess.run(
        [
            str(builder_python),
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--interpreter",
            str(install_python),
            "--manifest-path",
            str(manifest),
            "--target-dir",
            str(build_root / "cargo-target"),
            "--out",
            str(wheel_dir),
        ],
        cwd=worktree,
        env=env,
        check=True,
        capture_output=True,
    )
    wheels = tuple(wheel_dir.glob("placementscript-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one PlacementScript wheel, found {wheels}")
    wheel = wheels[0]
    wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(install_python),
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        cwd=worktree,
        env=env,
        check=True,
        capture_output=True,
    )
    return wheel_sha256


def _execution_attestation(
    worktree: Path,
    *,
    python: Path,
    wheel_sha256: str,
    env: dict[str, str],
    dependency_groups: tuple[str, ...] = (),
    dependency_extras: tuple[str, ...] = (),
    entrypoint_dependency_imports: tuple[str, ...] = (),
) -> dict[str, object]:
    """Attest the actual fresh-worktree interpreter, tools, lock, and native module."""
    maturin = python.parent / ("maturin.exe" if os.name == "nt" else "maturin")
    if not maturin.is_file():
        raise RuntimeError("fresh worktree environment does not contain Maturin")
    runtime = json.loads(
        _checked_output([str(python), "-c", _RUNTIME_ATTESTATION_PROBE], cwd=worktree, env=env)
    )
    runtime.update(
        {
            "uv_version": _checked_output(["uv", "--version"], cwd=worktree, env=env),
            "maturin_version": _checked_output([str(maturin), "--version"], cwd=worktree, env=env),
            "rustc_version": _checked_output(["rustc", "--version"], cwd=worktree, env=env),
            "cargo_version": _checked_output(["cargo", "--version"], cwd=worktree, env=env),
            "worktree_uv_lock_sha256": hashlib.sha256(
                (worktree / "uv.lock").read_bytes()
            ).hexdigest(),
            "runtime_cargo_lock_sha256": hashlib.sha256(
                (worktree / "runtime" / "Cargo.lock").read_bytes()
            ).hexdigest(),
            "placementscript_wheel_sha256": wheel_sha256,
            "dependency_groups": list(dependency_groups),
            "dependency_extras": list(dependency_extras),
            "entrypoint_dependency_imports": list(entrypoint_dependency_imports),
        }
    )
    return runtime


def validate_execution_attestation(
    registration: Mapping[str, object], attestation: Mapping[str, object]
) -> None:
    """Compare actual prepared-worktree facts with every P2 execution pin (R4-5/6)."""
    expected = registration["execution_environment"]
    if not isinstance(expected, Mapping):  # already checked; defensive at mutation boundary
        raise GateRefused("execution_environment is not a mapping")
    for key in sorted(_EXECUTION_PIN_KEYS):
        if attestation.get(key) != expected.get(key):
            raise GateRefused(
                f"prepared execution {key}={attestation.get(key)!r} differs from "
                f"P2 pin {expected.get(key)!r}"
            )
    if attestation.get("worktree_uv_lock_sha256") != registration["uv_lock_sha256"]:
        raise GateRefused("the detached worktree uv.lock actually used does not match the P2 hash")
    for key in (
        "dependency_groups",
        "dependency_extras",
        "entrypoint_dependency_imports",
    ):
        if attestation.get(key) != registration[key]:
            raise GateRefused(
                f"prepared execution {key}={attestation.get(key)!r} differs from "
                f"P2 registration {registration[key]!r}"
            )
    recomputed = _installed_distributions_sha256(
        attestation.get("installed_distributions")
    )
    if recomputed != attestation.get("installed_distributions_sha256"):
        raise GateRefused(
            "installed distribution names+versions do not match their attested hash"
        )


@contextlib.contextmanager
def prepare_pinned_execution(
    commit: str,
    editable_deps: tuple[EditableDepState, ...],
    execution_pins: Mapping[str, str],
    dependency_groups: tuple[str, ...],
    dependency_extras: tuple[str, ...],
    entrypoint_dependency_imports: tuple[str, ...],
    *,
    repo: Path = REPO,
) -> Iterator[PreparedExecution]:
    """Prepare a detached tree plus detached snapshots of external editables.

    The temporary layout preserves each relative path in ``uv.lock`` while ensuring it
    resolves to a worktree at the registered commit, never the live sibling checkout
    (R4-6). Environment sync, native-wheel build/install, and real import attestation
    all complete before this context yields, hence before the run-once marker.
    """
    root = Path(tempfile.mkdtemp(prefix="continuum-holdout-"))
    worktree = root / repo.name
    dependency_worktrees: list[tuple[Path, Path]] = []
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    env["UV_CACHE_DIR"] = str(root / "uv-cache")
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        states = {dep.name: dep for dep in editable_deps}
        pinned_targets = editable_out_of_repo_deps(worktree)
        if {name for name, _ in pinned_targets} != set(states):
            raise RuntimeError("pinned worktree editable dependency closure changed")
        for name, target in pinned_targets:
            state = states[name]
            if state.commit is None:
                raise RuntimeError(f"editable dependency {name} has no registered commit")
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    state.path,
                    "worktree",
                    "add",
                    "--detach",
                    str(target),
                    state.commit,
                ],
                check=True,
                capture_output=True,
            )
            dependency_worktrees.append((Path(state.path), target))

        subprocess.run(
            _uv_sync_command(
                execution_pins["python_version"],
                dependency_groups,
                dependency_extras,
            ),
            cwd=worktree,
            env=env,
            check=True,
            capture_output=True,
        )
        python = worktree / ".venv" / "bin" / "python"
        wheel_sha256 = _build_and_install_native_runtime(
            worktree,
            builder_python=python,
            install_python=python,
            build_root=root / "native-build",
            env=env,
        )
        attestation = _execution_attestation(
            worktree,
            python=python,
            wheel_sha256=wheel_sha256,
            env=env,
            dependency_groups=dependency_groups,
            dependency_extras=dependency_extras,
            entrypoint_dependency_imports=entrypoint_dependency_imports,
        )
        yield PreparedExecution(root, worktree, python, attestation)
    finally:
        for source, target in reversed(dependency_worktrees):
            subprocess.run(
                ["git", "-C", str(source), "worktree", "remove", "--force", str(target)],
                capture_output=True,
            )
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                capture_output=True,
            )
        shutil.rmtree(root, ignore_errors=True)


def preflight_prepared_entrypoint(
    prepared: PreparedExecution,
    module_name: str,
    func_name: str,
    dependency_imports: tuple[str, ...],
) -> None:
    """Import the P2-enumerated dependency closure and entrypoint before the marker."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(prepared.worktree)
    env.pop("VIRTUAL_ENV", None)
    try:
        subprocess.run(
            [
                str(prepared.python),
                "-c",
                _ENTRYPOINT_PREFLIGHT,
                str(prepared.worktree),
                module_name,
                func_name,
                json.dumps(dependency_imports, allow_nan=False),
            ],
            cwd=prepared.worktree,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip().splitlines()
        cause = detail[-1] if detail else f"subprocess exited {exc.returncode}"
        raise GateRefused(
            "prepared environment cannot import the registered entrypoint "
            f"dependency closure before the one-shot marker: {cause}"
        ) from exc


def _run_prepared_worktree(prepared: PreparedExecution, module_name: str, func_name: str) -> object:
    """Execute the registered entrypoint only after environment attestation succeeds."""
    with tempfile.TemporaryDirectory(prefix="holdout-gate-") as tmp:
        out_path = Path(tmp) / "result.json"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(prepared.worktree)
        env.pop("VIRTUAL_ENV", None)
        subprocess.run(
            [
                str(prepared.python),
                "-c",
                _WORKTREE_RUNNER,
                str(prepared.worktree),
                module_name,
                func_name,
                str(out_path),
            ],
            cwd=prepared.worktree,
            env=env,
            check=True,
        )
        return json.loads(out_path.read_text())


def main() -> int:
    driver_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    env = gather_environment()
    try:
        reg = check_gate(env, driver_sha256=driver_sha)
        listed = bundle_list_sha256()
        if listed != reg["bundle_list_sha256"]:
            raise GateRefused(
                "held-out bundle definitions do not match the registered hash "
                f"({listed[:12]} != {str(reg['bundle_list_sha256'])[:12]})"
            )
        module_name = str(reg["entrypoint"]).split(":")[0]
        verify_entrypoint_pinned(module_name, str(reg["entrypoint_sha256"]))
    except GateRefused as exc:
        print(f"GATE REFUSED: {exc}", file=sys.stderr)
        return 2

    # R4-5/R4-6: reconstruct, build, import, and compare the *actual* detached
    # environment before burning the one-shot marker. External editables are detached
    # snapshots at registered commits, never live paths.
    execution_pins = reg["execution_environment"]
    assert isinstance(execution_pins, dict)  # check_gate established this shape
    dependency_groups = tuple(reg["dependency_groups"])
    dependency_extras = tuple(reg["dependency_extras"])
    dependency_imports = tuple(reg["entrypoint_dependency_imports"])
    with prepare_pinned_execution(
        str(reg["pinned_commit"]),
        env.editable_deps,
        execution_pins,
        dependency_groups,
        dependency_extras,
        dependency_imports,
    ) as prepared:
        try:
            validate_execution_attestation(reg, prepared.attestation)
            module_name, func_name = str(reg["entrypoint"]).split(":")
            preflight_prepared_entrypoint(
                prepared, module_name, func_name, dependency_imports
            )
        except GateRefused as exc:
            print(f"GATE REFUSED: {exc}", file=sys.stderr)
            return 2

        _atomic_create(
            ATTEMPT,
            {
                "commit": env.head_commit,
                "tag": env.exact_tag,
                "driver_sha256": driver_sha,
                "editable_dependencies": {
                    d.name: {"source_path": d.path, "commit": d.commit} for d in env.editable_deps
                },
                "execution_attestation": prepared.attestation,
                "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            },
        )

        result = _run_prepared_worktree(prepared, module_name, func_name)
        RESULT.write_text(
            result_json(
                {
                    "registration": reg,
                    "attempt": json.loads(ATTEMPT.read_text()),
                    "result": result,
                    "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                }
            )
        )
    print(f"held-out evaluation complete; result written to {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
