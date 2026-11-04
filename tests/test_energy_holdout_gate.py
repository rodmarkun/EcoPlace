"""Run-once held-out driver tests (TODO 3.3): every refusal path of the access rule
(tagged commit, registration, marker, hash pins, confirmation), the O_EXCL marker
semantics, the R2-4 dependency-closure and import-free entrypoint-resolution
regressions, and the repository invariant that the gate has never been run."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sim.energy.holdout_gate import (
    ATTEMPT,
    REGISTRATION,
    RESULT,
    EditableDepState,
    GateEnvironment,
    GateRefused,
    PreparedExecution,
    _atomic_create,
    _build_and_install_native_runtime,
    _execution_attestation,
    _uv_sync_command,
    bundle_list_sha256,
    check_gate,
    editable_out_of_repo_deps,
    gather_environment,
    preflight_prepared_entrypoint,
    resolve_module_path,
    result_json,
    validate_execution_attestation,
)

DRIVER_SHA = "d" * 64
LOCK_SHA = "f" * 64
INSTALLED_DISTRIBUTIONS = [
    {"name": "evolved-continuum", "version": "0.0.1"},
    {"name": "gymnasium", "version": "1.3.0"},
]
INSTALLED_DISTRIBUTIONS_SHA = hashlib.sha256(
    json.dumps(
        INSTALLED_DISTRIBUTIONS,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
).hexdigest()
EXECUTION_PINS = {
    "python_version": "3.12.4",
    "python_implementation": "CPython",
    "python_platform": "macosx-15.0-arm64",
    "machine": "arm64",
    "system": "Darwin",
    "uv_version": "uv 0.10.2",
    "maturin_version": "maturin 1.14.1",
    "rustc_version": "rustc 1.93.0",
    "cargo_version": "cargo 1.93.0",
    "installed_distributions_sha256": INSTALLED_DISTRIBUTIONS_SHA,
}


def good_env(**overrides) -> GateEnvironment:
    reg = {
        "pinned_commit": "c" * 40,
        "driver_sha256": DRIVER_SHA,
        "bundle_list_sha256": "b" * 64,
        "entrypoint": "evolution.holdout:evaluate_frozen_champions",
        "entrypoint_sha256": "e" * 64,
        "one_shot_assumption": "honest-operator-local-marker",
        "editable_dependencies": {"shinka": "a" * 40},
        "dependency_groups": ["dev"],
        "dependency_extras": ["milp", "rl"],
        "entrypoint_dependency_imports": [
            "gymnasium",
            "highspy",
            "sb3_contrib",
            "stable_baselines3",
            "torch",
        ],
        "uv_lock_sha256": LOCK_SHA,
        "execution_environment": EXECUTION_PINS,
    }
    reg_bytes = json.dumps(reg, sort_keys=True).encode()
    kwargs = dict(
        head_commit="c" * 40,
        exact_tag="v1.0-p2-freeze",
        dirty_entries=(),
        registration_exists=True,
        registration=reg,
        registration_head_bytes=reg_bytes,
        registration_disk_bytes=reg_bytes,
        attempt_marker_exists=False,
        confirm_env="1",
        editable_deps=(
            EditableDepState(
                name="shinka", path="/elsewhere/ShinkaEvolve", commit="a" * 40, dirty=False
            ),
        ),
    )
    kwargs.update(overrides)
    return GateEnvironment(**kwargs)


class TestRefusals:
    def test_happy_path_returns_registration(self):
        reg = check_gate(good_env(), driver_sha256=DRIVER_SHA)
        assert reg["entrypoint"] == "evolution.holdout:evaluate_frozen_champions"

    def test_refuses_when_marker_exists(self):
        with pytest.raises(GateRefused, match="run-once"):
            check_gate(good_env(attempt_marker_exists=True), driver_sha256=DRIVER_SHA)

    def test_refuses_without_registration(self):
        with pytest.raises(GateRefused, match="P2_HELD_OUT_BUNDLES"):
            check_gate(
                good_env(registration_exists=False, registration=None),
                driver_sha256=DRIVER_SHA,
            )

    def test_refuses_registration_missing_keys(self):
        env = good_env()
        for key in (
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
        ):
            broken = dict(env.registration)
            del broken[key]
            with pytest.raises(GateRefused, match=key):
                check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("driver_sha256", "g" * 64),
            ("bundle_list_sha256", "short"),
            ("entrypoint_sha256", "-" * 64),
            ("uv_lock_sha256", "not-a-sha"),
            ("pinned_commit", "z" * 40),
        ],
    )
    def test_refuses_malformed_registered_digests(self, key, bad):
        broken = dict(good_env().registration)
        broken[key] = bad
        with pytest.raises(GateRefused, match=key):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_refuses_malformed_editable_commit_digest(self):
        broken = dict(good_env().registration)
        broken["editable_dependencies"] = {"shinka": "not-a-commit"}
        with pytest.raises(GateRefused, match="editable_dependencies"):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    @pytest.mark.parametrize(
        ("key", "bad"),
        [
            ("dependency_groups", "dev"),
            ("dependency_groups", ["dev", "dev"]),
            ("dependency_extras", ["rl", "milp"]),
            ("entrypoint_dependency_imports", []),
            ("entrypoint_dependency_imports", ["bad/import"]),
        ],
    )
    def test_refuses_malformed_dependency_selection(self, key, bad):
        broken = dict(good_env().registration)
        broken[key] = bad
        with pytest.raises(GateRefused, match=key):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_refuses_malformed_distribution_set_pin(self):
        broken = dict(good_env().registration)
        broken["execution_environment"] = {
            **EXECUTION_PINS,
            "installed_distributions_sha256": "not-a-sha",
        }
        with pytest.raises(GateRefused, match="installed_distributions_sha256"):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_refuses_registration_not_committed_at_head(self):
        """R1-8: an uncommitted registration must not be honored."""
        with pytest.raises(GateRefused, match="not committed at HEAD"):
            check_gate(good_env(registration_head_bytes=None), driver_sha256=DRIVER_SHA)

    def test_refuses_registration_differing_from_head_blob(self):
        """R1-8: a dirty registration cannot redirect execution while HEAD matches."""
        env = good_env()
        swapped = dict(env.registration)
        swapped["entrypoint"] = "attacker.module:run"
        with pytest.raises(GateRefused, match="differs from the registration committed"):
            check_gate(
                good_env(
                    registration=swapped,
                    registration_disk_bytes=json.dumps(swapped, sort_keys=True).encode(),
                ),
                driver_sha256=DRIVER_SHA,
            )

    def test_refuses_missing_one_shot_assumption(self):
        broken = dict(good_env().registration)
        del broken["one_shot_assumption"]
        with pytest.raises(GateRefused, match="honest-operator"):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_refuses_untagged_head(self):
        with pytest.raises(GateRefused, match="not at a tag"):
            check_gate(good_env(exact_tag=None), driver_sha256=DRIVER_SHA)

    def test_refuses_commit_mismatch(self):
        with pytest.raises(GateRefused, match="!= registered commit"):
            check_gate(good_env(head_commit="e" * 40), driver_sha256=DRIVER_SHA)

    def test_refuses_dirty_sources(self):
        with pytest.raises(GateRefused, match="dirty/untracked"):
            check_gate(
                good_env(dirty_entries=(" M sim/energy/bundles.py",)),
                driver_sha256=DRIVER_SHA,
            )

    def test_refuses_driver_hash_mismatch(self):
        with pytest.raises(GateRefused, match="driver's hash"):
            check_gate(good_env(), driver_sha256="0" * 64)

    def test_refuses_without_confirmation(self):
        with pytest.raises(GateRefused, match="CONFIRM_HELD_OUT_GATE"):
            check_gate(good_env(confirm_env=None), driver_sha256=DRIVER_SHA)

    def test_marker_check_precedes_everything(self):
        """A burned attempt refuses even if everything else is also wrong — the
        second casual invocation must fail on the marker, loudly."""
        with pytest.raises(GateRefused, match="run-once"):
            check_gate(
                good_env(attempt_marker_exists=True, exact_tag=None, confirm_env=None),
                driver_sha256="0" * 64,
            )


class TestMarker:
    def test_atomic_create_refuses_second_invocation(self, tmp_path: Path):
        marker = tmp_path / "HOLDOUT_ATTEMPT.json"
        _atomic_create(marker, {"n": 1})
        assert json.loads(marker.read_text()) == {"n": 1}
        with pytest.raises(FileExistsError):
            _atomic_create(marker, {"n": 2})
        assert json.loads(marker.read_text()) == {"n": 1}  # first attempt retained

    def test_marker_is_read_only(self, tmp_path: Path):
        marker = tmp_path / "m.json"
        _atomic_create(marker, {})
        assert (marker.stat().st_mode & 0o777) == 0o444


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def scratch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "prereg").mkdir(parents=True)
    (repo / "prereg" / "README.md").write_text("scratch\n")
    (repo / "sim").mkdir()
    (repo / "sim" / "core.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@test")
    _git(repo, "config", "user.name", "test")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


class TestDependencyClosure:
    """R2-4 adversarial regressions: the environment gathering must report dirt
    anywhere in the repository (the old path allowlist let a dirty ``evolution/``
    scenario or baseline module through), and entrypoint resolution must never
    import working-tree code."""

    def test_clean_scratch_repo_reports_no_dirt(self, scratch_repo):
        env = gather_environment(scratch_repo)
        assert env.dirty_entries == ()
        assert env.registration_exists is False
        assert env.registration_head_bytes is None

    def test_dirt_outside_the_old_allowlist_is_reported(self, scratch_repo):
        """The demonstrated bypass: an untracked evolution/ module (imported by any
        realistic entrypoint) was previously invisible to the gate."""
        (scratch_repo / "evolution").mkdir()
        (scratch_repo / "evolution" / "stray.py").write_text("EVIL = True\n")
        (scratch_repo / "baselines").mkdir()
        (scratch_repo / "baselines" / "tweak.py").write_text("x = 2\n")
        env = gather_environment(scratch_repo)
        joined = "\n".join(env.dirty_entries)
        assert "evolution/stray.py" in joined
        assert "baselines/tweak.py" in joined
        with pytest.raises(GateRefused, match="entire repository"):
            check_gate(
                GateEnvironment(
                    **{
                        **good_env().__dict__,
                        "dirty_entries": env.dirty_entries,
                    }
                ),
                driver_sha256=DRIVER_SHA,
            )

    def test_modified_tracked_file_anywhere_is_reported(self, scratch_repo):
        (scratch_repo / "sim" / "core.py").write_text("x = 999\n")
        env = gather_environment(scratch_repo)
        assert any("sim/core.py" in e for e in env.dirty_entries)


class TestImportFreeEntrypointResolution:
    """R2-4: the registered entrypoint is resolved textually — parent packages are
    never executed during gate checks."""

    def test_resolution_is_textual_and_never_imports(self, tmp_path: Path):
        pkg = tmp_path / "somepkg"
        pkg.mkdir()
        marker = tmp_path / "IMPORTED_MARKER"
        (pkg / "__init__.py").write_text(
            f"open({str(marker)!r}, 'w').write('imported')\n"
        )
        (pkg / "mod.py").write_text("def f(bundles):\n    return {}\n")
        path = resolve_module_path("somepkg.mod", tmp_path)
        assert path == (pkg / "mod.py").resolve()
        assert not marker.exists()  # the parent package was never executed

    def test_package_init_layout_resolves(self, tmp_path: Path):
        pkg = tmp_path / "pkg" / "sub"
        pkg.mkdir(parents=True)
        (tmp_path / "pkg" / "__init__.py").write_text("")
        (pkg / "__init__.py").write_text("")
        assert resolve_module_path("pkg.sub", tmp_path) == (pkg / "__init__.py").resolve()

    def test_ambiguous_module_layout_refused(self, tmp_path: Path):
        pkg = tmp_path / "amb"
        (pkg / "mod").mkdir(parents=True)
        (pkg / "mod.py").write_text("")
        (pkg / "mod" / "__init__.py").write_text("")
        with pytest.raises(GateRefused, match="ambiguous"):
            resolve_module_path("amb.mod", tmp_path)

    def test_nonsense_module_names_rejected(self, tmp_path: Path):
        with pytest.raises(GateRefused, match="dotted module"):
            resolve_module_path("evil/../../etc/passwd", tmp_path)
        with pytest.raises(GateRefused, match="dotted module"):
            resolve_module_path("", tmp_path)
        with pytest.raises(GateRefused, match="does not resolve"):
            resolve_module_path("missing.module", tmp_path)


class TestEnvironmentClosure:
    """R3-3: out-of-repository editable dependencies (results-relevant code the
    pinned commit cannot contain) must be registered, clean, and at the registered
    commit; the evaluation runs in a frozen-synced environment of the pinned tree."""

    def test_unregistered_editable_dependency_refused(self):
        env = good_env()
        broken = dict(env.registration)
        broken["editable_dependencies"] = {}
        with pytest.raises(GateRefused, match="unregistered editable source|diverge"):
            check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_dirty_editable_dependency_refused(self):
        deps = (
            EditableDepState(
                name="shinka", path="/elsewhere/ShinkaEvolve", commit="a" * 40, dirty=True
            ),
        )
        with pytest.raises(GateRefused, match="uncommitted changes"):
            check_gate(good_env(editable_deps=deps), driver_sha256=DRIVER_SHA)

    def test_editable_dependency_commit_mismatch_refused(self):
        deps = (
            EditableDepState(
                name="shinka", path="/elsewhere/ShinkaEvolve", commit="f" * 40, dirty=False
            ),
        )
        with pytest.raises(GateRefused, match="registration pins"):
            check_gate(good_env(editable_deps=deps), driver_sha256=DRIVER_SHA)

    def test_non_git_editable_dependency_refused(self):
        deps = (
            EditableDepState(
                name="shinka", path="/elsewhere/ShinkaEvolve", commit=None, dirty=False
            ),
        )
        with pytest.raises(GateRefused, match="not a git"):
            check_gate(good_env(editable_deps=deps), driver_sha256=DRIVER_SHA)

    def test_lockfile_parsing_finds_out_of_repo_editables(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "uv.lock").write_text(
            'version = 1\n'
            '[[package]]\nname = "inrepo"\nversion = "0.1"\n'
            'source = { editable = "pkgs/inrepo" }\n'
            '[[package]]\nname = "shinka"\nversion = "0.1"\n'
            'source = { editable = "../ShinkaEvolve" }\n'
            '[[package]]\nname = "numpy"\nversion = "2.0"\n'
            'source = { registry = "https://pypi.org/simple" }\n'
        )
        deps = editable_out_of_repo_deps(repo)
        assert [name for name, _ in deps] == ["shinka"]
        assert deps[0][1] == (tmp_path / "ShinkaEvolve").resolve()

    def test_real_repo_editable_closure_is_shinka_only(self):
        from sim.energy.holdout_gate import REPO

        names = [name for name, _ in editable_out_of_repo_deps(REPO)]
        assert names == ["shinka-evolve"], (
            "the out-of-repo editable closure changed — update the gate "
            f"registration requirements: {names}"
        )

    def test_dirty_repository_not_erased_by_clean_editable_dependency(
        self, scratch_repo, tmp_path
    ):
        """R4-6 exact counterexample: dependency dirt uses a distinct variable and
        cannot overwrite the repository's dirty-entry tuple."""
        dependency = tmp_path / "external-dep"
        dependency.mkdir()
        (dependency / "pyproject.toml").write_text("[project]\nname='dep'\nversion='0.1'\n")
        _git(dependency, "init", "-q")
        _git(dependency, "config", "user.email", "test@test")
        _git(dependency, "config", "user.name", "test")
        _git(dependency, "add", "-A")
        _git(dependency, "commit", "-qm", "init")
        (scratch_repo / "uv.lock").write_text(
            'version = 1\n[[package]]\nname = "dep"\nversion = "0.1"\n'
            'source = { editable = "../external-dep" }\n'
        )
        _git(scratch_repo, "add", "uv.lock")
        _git(scratch_repo, "commit", "-qm", "lock")
        (scratch_repo / "sim" / "core.py").write_text("x = 999\n")
        env = gather_environment(scratch_repo)
        assert env.editable_deps[0].dirty is False
        assert any("sim/core.py" in item for item in env.dirty_entries)

    def test_actual_execution_attestation_must_match_every_pin(self):
        reg = good_env().registration
        actual = {
            **EXECUTION_PINS,
            "worktree_uv_lock_sha256": LOCK_SHA,
            "dependency_groups": reg["dependency_groups"],
            "dependency_extras": reg["dependency_extras"],
            "entrypoint_dependency_imports": reg["entrypoint_dependency_imports"],
            "installed_distributions": INSTALLED_DISTRIBUTIONS,
        }
        validate_execution_attestation(reg, actual)
        actual["rustc_version"] = "rustc surprise"
        with pytest.raises(GateRefused, match="rustc_version"):
            validate_execution_attestation(reg, actual)
        actual = {
            **actual,
            "rustc_version": EXECUTION_PINS["rustc_version"],
            "worktree_uv_lock_sha256": "0" * 64,
        }
        with pytest.raises(GateRefused, match="uv.lock actually used"):
            validate_execution_attestation(reg, actual)

        actual = {**actual, "worktree_uv_lock_sha256": LOCK_SHA}
        actual["installed_distributions"] = [
            *INSTALLED_DISTRIBUTIONS,
            {"name": "torch", "version": "999"},
        ]
        with pytest.raises(GateRefused, match="names\\+versions"):
            validate_execution_attestation(reg, actual)

    def test_frozen_sync_uses_exact_registered_groups_and_extras(self):
        command = _uv_sync_command("3.12.4", ("dev",), ("milp", "rl"))
        assert command == [
            "uv",
            "sync",
            "--frozen",
            "--no-default-groups",
            "--python",
            "3.12.4",
            "--group",
            "dev",
            "--extra",
            "milp",
            "--extra",
            "rl",
        ]

    def test_entrypoint_dependency_import_preflight_fails_closed(self, tmp_path):
        (tmp_path / "registered_entrypoint.py").write_text(
            "def run(bundles):\n    return len(bundles)\n"
        )
        prepared = PreparedExecution(
            root=tmp_path,
            worktree=tmp_path,
            python=Path(sys.executable),
            attestation={},
        )
        preflight_prepared_entrypoint(
            prepared,
            "registered_entrypoint",
            "run",
            ("json",),
        )
        with pytest.raises(GateRefused, match="cannot import"):
            preflight_prepared_entrypoint(
                prepared,
                "registered_entrypoint",
                "run",
                ("definitely_missing_phase3_dependency",),
            )

    def test_main_refuses_failed_preflight_before_attempt_marker(
        self, monkeypatch, tmp_path
    ):
        """R5-4 reproduction path: missing PPO imports cannot burn the marker."""
        import sim.energy.holdout_gate as gate

        reg = good_env().registration
        prepared = PreparedExecution(
            root=tmp_path,
            worktree=tmp_path,
            python=Path(sys.executable),
            attestation={},
        )

        @contextlib.contextmanager
        def fake_prepare(*_args, **_kwargs):
            yield prepared

        marker_called = False

        def marker_must_not_run(*_args, **_kwargs):
            nonlocal marker_called
            marker_called = True

        monkeypatch.setattr(gate, "gather_environment", good_env)
        monkeypatch.setattr(gate, "check_gate", lambda *_args, **_kwargs: reg)
        monkeypatch.setattr(
            gate, "bundle_list_sha256", lambda: reg["bundle_list_sha256"]
        )
        monkeypatch.setattr(gate, "verify_entrypoint_pinned", lambda *_args: None)
        monkeypatch.setattr(gate, "prepare_pinned_execution", fake_prepare)
        monkeypatch.setattr(
            gate, "validate_execution_attestation", lambda *_args: None
        )
        monkeypatch.setattr(
            gate,
            "preflight_prepared_entrypoint",
            lambda *_args: (_ for _ in ()).throw(
                GateRefused("missing registered PPO dependency")
            ),
        )
        monkeypatch.setattr(gate, "_atomic_create", marker_must_not_run)

        assert gate.main() == 2
        assert marker_called is False

    def test_native_runtime_builds_installs_and_imports_in_fresh_environment(self, tmp_path):
        """R4-5 load-bearing integration: from a real detached git worktree, build a
        locked PlacementScript wheel, install it into a new empty venv, import its
        native API, and hash both artifacts."""
        from sim.energy.holdout_gate import REPO

        source = tmp_path / "native-source"
        source.mkdir()
        tracked_runtime = subprocess.run(
            ["git", "ls-files", "runtime"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        assert tracked_runtime
        for name in (*tracked_runtime, "uv.lock"):
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
        _git(source, "init", "-q")
        _git(source, "config", "user.email", "test@test")
        _git(source, "config", "user.name", "test")
        _git(source, "add", "-A")
        _git(source, "commit", "-qm", "pinned native source")
        detached = tmp_path / "detached-native"
        _git(source, "worktree", "add", "--detach", str(detached), "HEAD")

        env = dict(os.environ)
        env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
        venv = tmp_path / "fresh-venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(venv)],
            env=env,
            check=True,
            capture_output=True,
        )
        python = venv / "bin" / "python"
        wheel_sha = _build_and_install_native_runtime(
            detached,
            builder_python=Path(sys.executable),
            install_python=python,
            build_root=tmp_path / "native-build",
            env=env,
        )
        # Production's frozen sync installs the lock-pinned Maturin executable into
        # this venv. The focused integration isolates native wheel construction/import,
        # so copy the already lock-pinned test runner executable into the fresh venv.
        shutil.copy2(Path(sys.executable).parent / "maturin", python.parent / "maturin")
        actual = _execution_attestation(
            detached, python=python, wheel_sha256=wheel_sha, env=env
        )
        assert len(actual["placementscript_wheel_sha256"]) == 64
        assert len(actual["placementscript_native_sha256"]) == 64
        assert len(actual["installed_distributions_sha256"]) == 64
        assert actual["installed_distributions"]
        assert actual["placementscript_native_name"].endswith((".so", ".pyd"))


class TestEntrypointShapeAndResultSerialization:
    """R3-5: a malformed entrypoint refuses before the run-once marker burns, and
    held-out results are serialized with NaN/Infinity forbidden."""

    def test_malformed_entrypoint_refused_before_marker(self):
        for bad in ("no_colon", "a:b:c", "mod:", ":fn", "1bad.mod:fn", "mod:not-an-ident"):
            broken = dict(good_env().registration)
            broken["entrypoint"] = bad
            with pytest.raises(GateRefused, match="entrypoint"):
                check_gate(good_env(registration=broken), driver_sha256=DRIVER_SHA)

    def test_result_serialization_refuses_nan(self):
        assert '"x": 1.5' in result_json({"x": 1.5})
        with pytest.raises(ValueError):
            result_json({"x": float("nan")})
        with pytest.raises(ValueError):
            result_json({"nested": [{"y": float("inf")}]})

    def test_worktree_runner_forbids_nan(self):
        from sim.energy.holdout_gate import _WORKTREE_RUNNER

        assert "allow_nan=False" in _WORKTREE_RUNNER
        assert "import placementscript" in _WORKTREE_RUNNER


class TestRepositoryInvariants:
    def test_gate_has_never_run_in_this_repo(self):
        """Phase 3 ships machinery only: no attempt marker, no result, and no P2
        registration may exist yet (P2 freezes Sep 8)."""
        assert not ATTEMPT.exists()
        assert not RESULT.exists()
        assert not REGISTRATION.exists()

    def test_bundle_list_hash_is_stable(self):
        assert bundle_list_sha256() == bundle_list_sha256()
        assert len(bundle_list_sha256()) == 64

    def test_pinned_data_files_are_visible_to_git(self):
        """R2-4 closure: a gitignored pinned-data directory would be invisible to
        the gate's whole-repo clean-tree check *and* absent from the pinned
        worktree the evaluation runs in. The pinned sources, the SPECpower table,
        and every retained artifact must be trackable (regression for the bare
        `data/` ignore pattern, now root-anchored)."""
        from sim.energy.holdout_gate import REPO
        from sim.energy.sources import DATA_DIR

        paths = [
            DATA_DIR / "pinned_sources.json",
            DATA_DIR / "specpower.json",
            *sorted((DATA_DIR / "artifacts").iterdir()),
        ]
        for path in paths:
            proc = subprocess.run(
                ["git", "check-ignore", str(path)], cwd=REPO, capture_output=True
            )
            assert proc.returncode != 0, f"{path} is gitignored - gate closure broken"
