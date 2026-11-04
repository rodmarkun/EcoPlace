"""Spec-layer tests: validation rules, distribution semantics, canonical JSON
round-trips, fixture integrity, and the T6 dev/acceptance disjointness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sim_helpers import DEV_FIXTURES, fanout_config

from analysis.tracking import spawn_rng
from sim.spec import (
    ArrivalSpec,
    BackendSpec,
    CallEdgeSpec,
    CallSpec,
    ClusterSpec,
    DistributionSpec,
    NodeClassSpec,
    NodeSpec,
    RebalancerParams,
    SchedulerParams,
    ServiceSpec,
    TemplateSpec,
    config_sha256,
    det,
    episode_config_from_json,
    episode_config_to_json,
    expo,
    load_episode_config,
)

ACCEPTANCE = DEV_FIXTURES.parent / "acceptance"


class TestDistributions:
    def test_means(self):
        assert det(0.5).mean() == 0.5
        assert expo(0.25).mean() == 0.25
        lognormal = DistributionSpec("lognormal", {"mu": -3.0, "sigma": 0.5})
        assert lognormal.mean() == pytest.approx(np.exp(-3.0 + 0.125))
        gamma = DistributionSpec("gamma", {"shape": 2.0, "scale": 0.05})
        assert gamma.mean() == pytest.approx(0.1)

    def test_sampling_is_seed_deterministic(self):
        d = expo(0.1)
        a = [d.sample(spawn_rng(7, "test", "a")) for _ in range(1)]
        b = [d.sample(spawn_rng(7, "test", "a")) for _ in range(1)]
        assert a == b

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown distribution"):
            DistributionSpec("uniform", {"low": 0.0})
        with pytest.raises(ValueError, match="requires params"):
            DistributionSpec("exponential", {"rate": 1.0})


class TestClusterValidation:
    def test_suspendable_requires_all_transition_parameters(self):
        partial = NodeClassSpec(
            name="half",
            power_idle_w=1.0,
            power_max_w=2.0,
            suspend_energy_j=1.0,  # missing latencies + wake energy
        )
        assert partial.suspendable is False
        with pytest.raises(ValueError, match="not suspendable"):
            ClusterSpec(
                classes=(partial,),
                nodes=(NodeSpec(0, "half", 2, 100, 100, initially_suspended=True),),
                latency_ms=((0.0,),),
            )

    def test_latency_matrix_must_be_symmetric_zero_diagonal(self):
        cls = NodeClassSpec(name="c", power_idle_w=1.0, power_max_w=2.0)
        nodes = (NodeSpec(0, "c", 2, 100, 100), NodeSpec(1, "c", 2, 100, 100))
        with pytest.raises(ValueError, match="symmetric"):
            ClusterSpec(classes=(cls,), nodes=nodes, latency_ms=((0.0, 1.0), (2.0, 0.0)))
        with pytest.raises(ValueError, match="diagonal"):
            ClusterSpec(classes=(cls,), nodes=nodes, latency_ms=((1.0, 1.0), (1.0, 0.0)))

    def test_dense_node_ids_enforced(self):
        cls = NodeClassSpec(name="c", power_idle_w=1.0, power_max_w=2.0)
        with pytest.raises(ValueError, match="dense"):
            ClusterSpec(
                classes=(cls,),
                nodes=(NodeSpec(1, "c", 2, 100, 100),),
                latency_ms=((0.0,),),
            )

    def test_cluster_size_bounds_enforced(self):
        """R1-12: the spec §4.1 host domain (1..512 nodes) is enforced at construction,
        so no legal episode can produce an out-of-domain ctx."""
        cls = NodeClassSpec(name="c", power_idle_w=1.0, power_max_w=2.0)
        with pytest.raises(ValueError, match="1..512"):
            ClusterSpec(classes=(cls,), nodes=(), latency_ms=())
        n = 513
        nodes = tuple(NodeSpec(i, "c", 2, 100, 100) for i in range(n))
        lat = tuple(tuple(0.0 if i == j else 1.0 for j in range(n)) for i in range(n))
        with pytest.raises(ValueError, match="1..512"):
            ClusterSpec(classes=(cls,), nodes=nodes, latency_ms=lat)

    def test_non_finite_latency_rejected(self):
        cls = NodeClassSpec(name="c", power_idle_w=1.0, power_max_w=2.0)
        nodes = (NodeSpec(0, "c", 2, 100, 100), NodeSpec(1, "c", 2, 100, 100))
        with pytest.raises(ValueError):
            ClusterSpec(
                classes=(cls,),
                nodes=nodes,
                latency_ms=((0.0, float("nan")), (float("nan"), 0.0)),
            )


class TestSpecImmutability:
    def test_distribution_params_are_frozen(self):
        d = expo(0.1)
        with pytest.raises(TypeError):
            d.params["mean"] = 99.0  # frozen mapping (R1-12)

    def test_non_finite_distribution_params_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            DistributionSpec("exponential", {"mean": float("inf")})
        with pytest.raises(ValueError, match="> 0"):
            DistributionSpec("gamma", {"shape": 0.0, "scale": 1.0})


class TestWorkloadValidation:
    def test_forward_edges_cannot_carry_timeouts(self):
        with pytest.raises(ValueError, match="forward"):
            CallSpec(
                service=0,
                mode="forward",
                children=(CallEdgeSpec(child=CallSpec(service=1), timeout_s=1.0),),
            )

    def test_backend_curve_must_ascend(self):
        with pytest.raises(ValueError, match="ascend"):
            BackendSpec("db", ((2.0, 0.1), (1.0, 0.2)))

    def test_arrival_validation(self):
        with pytest.raises(ValueError, match="rate_per_s"):
            ArrivalSpec("poisson", 0.0)
        with pytest.raises(ValueError, match="kind"):
            ArrivalSpec("bursty", 1.0)

    def test_nan_and_fractional_counts_are_rejected(self):
        service = fanout_config().workload.services[0]
        with pytest.raises(ValueError, match="cpu_request"):
            ServiceSpec(**{**service.__dict__, "cpu_request": float("nan")})
        with pytest.raises(ValueError, match="max_retries"):
            CallEdgeSpec(child=CallSpec(service=0), max_retries=1.5)
        with pytest.raises(ValueError, match="budget_per_window"):
            RebalancerParams(budget_per_window=1.5)
        with pytest.raises(ValueError, match="slo_latency_s"):
            TemplateSpec(
                id=0,
                name="nan-slo",
                root=CallSpec(service=0),
                arrival=ArrivalSpec("deterministic", 1.0),
                slo_latency_s=float("nan"),
            )

    def test_truth_capacity_check_cannot_be_disabled(self):
        with pytest.raises(ValueError, match="must stay enabled"):
            SchedulerParams(bind_time_capacity_check=False)

    def test_placementscript_shape_domain_is_enforced_at_spec_construction(self):
        """R4-3: every legal episode remains inside the native host shape domain."""
        from dataclasses import replace

        from sim.spec import MAX_GRAPH_EDGES, MAX_PODS, WorkloadSpec

        base = fanout_config().workload.services[0]
        template = TemplateSpec(
            0,
            "single",
            CallSpec(0),
            ArrivalSpec("deterministic", 1.0),
        )
        at_pod_limit = replace(base, initial_replicas=1, max_replicas=MAX_PODS)
        WorkloadSpec((at_pod_limit,), (template,))
        with pytest.raises(ValueError, match=r"sum\(max_replicas\)"):
            WorkloadSpec(
                (replace(at_pod_limit, max_replicas=MAX_PODS + 1),),
                (template,),
            )

        left = replace(base, id=0, initial_replicas=1, max_replicas=256)
        right = replace(base, id=1, name="right", initial_replicas=1, max_replicas=256)
        edge_template = TemplateSpec(
            0,
            "edge-limit",
            CallSpec(0, (CallEdgeSpec(CallSpec(1)),)),
            ArrivalSpec("deterministic", 1.0),
        )
        exact = WorkloadSpec((left, right), (edge_template,))
        assert left.max_replicas * right.max_replicas == MAX_GRAPH_EDGES
        assert exact.services[0].max_replicas == 256
        with pytest.raises(ValueError, match="expanded pod graph"):
            WorkloadSpec(
                (replace(left, max_replicas=257), right),
                (edge_template,),
            )

    def test_placementscript_scalar_domain_limits_are_enforced(self):
        """R4-3: max+1 probes fail before a native policy sees the episode."""
        from dataclasses import replace

        from sim.spec import (
            MAX_NODE_CAPACITY,
            MAX_NOW_MS,
            MAX_PAYLOAD_KB,
            MAX_PROPOSAL_BUDGET,
            MAX_RATE,
            MAX_RESOURCE,
            MAX_WATTS,
            NodeClassSpec,
            NodeSpec,
        )

        with pytest.raises(ValueError, match="power_idle_w"):
            NodeClassSpec("too-hot", MAX_WATTS + 1, MAX_WATTS + 1)
        NodeSpec(0, "n", 2, MAX_NODE_CAPACITY, MAX_NODE_CAPACITY)
        with pytest.raises(ValueError, match="cpu_capacity"):
            NodeSpec(0, "n", 2, MAX_NODE_CAPACITY + 1, MAX_NODE_CAPACITY)
        service = fanout_config().workload.services[0]
        with pytest.raises(ValueError, match="image_mb"):
            replace(service, image_mb=MAX_RESOURCE + 1)
        with pytest.raises(ValueError, match="rate_per_s"):
            ArrivalSpec("deterministic", MAX_RATE + 1)
        with pytest.raises(ValueError, match=r"request_kb \+ response_kb"):
            CallEdgeSpec(
                CallSpec(0),
                request_kb=MAX_PAYLOAD_KB,
                response_kb=1,
            )
        with pytest.raises(ValueError, match="budget_per_window"):
            RebalancerParams(budget_per_window=MAX_PROPOSAL_BUDGET + 1)
        config = fanout_config()
        with pytest.raises(ValueError, match="now_ms"):
            replace(config, horizon_s=MAX_NOW_MS / 1000.0 + 1.0)

    def test_float_typed_programmatic_and_json_values_are_normalized(self):
        """R4-2: integer-valued reals become JSON/native floats, never wire ints."""
        from sim.spec import NodeClassSpec

        node_class = NodeClassSpec("integer-input", 1, 2)
        edge = CallEdgeSpec(CallSpec(0), request_kb=1, response_kb=2)
        arrival = ArrivalSpec("deterministic", 3, start_s=4)
        assert type(node_class.power_idle_w) is float
        assert type(node_class.power_max_w) is float
        assert type(edge.request_kb) is float
        assert type(edge.response_kb) is float
        assert type(arrival.rate_per_s) is float
        assert type(arrival.start_s) is float

        raw = json.loads((DEV_FIXTURES / "small_cluster.json").read_text())
        raw["cluster"]["classes"][0]["power_idle_w"] = 3
        raw["cluster"]["classes"][0]["power_max_w"] = 8
        loaded = episode_config_from_json(json.dumps(raw))
        assert type(loaded.cluster.classes[0].power_idle_w) is float
        assert type(loaded.cluster.classes[0].power_max_w) is float


class TestJsonRoundTrip:
    def test_round_trip_preserves_hash(self):
        config = fanout_config(horizon_s=42.0)
        text = episode_config_to_json(config)
        loaded = episode_config_from_json(text)
        assert config_sha256(loaded) == config_sha256(config)
        assert loaded == config

    def test_all_dev_fixtures_load_and_hash_stably(self):
        for path in sorted(DEV_FIXTURES.glob("*.json")):
            config = load_episode_config(path)
            again = episode_config_from_json(episode_config_to_json(config))
            assert config_sha256(again) == config_sha256(config), path.name

    def test_every_numeric_fixture_leaf_rejects_nan(self):
        """R2-4: every numeric leaf entering through JSON is finite-validated."""
        original = json.loads((DEV_FIXTURES / "small_cluster.json").read_text())

        def numeric_paths(value, path=()):
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, int | float):
                yield path
                return
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from numeric_paths(child, (*path, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    yield from numeric_paths(child, (*path, index))

        def replace_at(value, path):
            clone = json.loads(json.dumps(value))
            cursor = clone
            for part in path[:-1]:
                cursor = cursor[part]
            cursor[path[-1]] = float("nan")
            return clone

        paths = list(numeric_paths(original))
        assert len(paths) > 30
        for path in paths:
            mutated = replace_at(original, path)
            with pytest.raises((ValueError, TypeError), match="finite|integer|tier"):
                episode_config_from_json(json.dumps(mutated))


class TestFixtureDiscipline:
    """T6: development and acceptance fixtures are disjoint; the acceptance set is
    frozen by P1 and never consumed by this suite (this test only inspects bytes)."""

    def test_directories_exist_and_acceptance_has_freeze_readme(self):
        assert DEV_FIXTURES.is_dir()
        assert ACCEPTANCE.is_dir()
        readme = (ACCEPTANCE / "README.md").read_text()
        assert "FROZEN AT P1" in readme
        assert "DO NOT RUN" in readme.upper()

    def test_acceptance_fixtures_differ_from_dev_counterparts(self):
        dev_files = {p.name: p for p in DEV_FIXTURES.glob("*.json")}
        acc_files = {p.name: p for p in ACCEPTANCE.glob("*.json")}
        assert set(dev_files) == set(acc_files)  # structurally analogous pairs
        for name, dev_path in dev_files.items():
            dev_cfg = load_episode_config(dev_path)
            acc_cfg = load_episode_config(acc_files[name])
            assert config_sha256(dev_cfg) != config_sha256(acc_cfg), (
                f"{name}: acceptance fixture must not duplicate the development one"
            )

    def test_no_sim_or_test_code_references_acceptance_paths(self):
        repo = Path(__file__).resolve().parent.parent
        offenders = []
        for path in [*sorted((repo / "sim").rglob("*.py")), *sorted((repo / "tests").glob("*.py"))]:
            # The generator and the one-shot gate driver are the only sanctioned
            # touchers of the acceptance set (R1-14).
            if path.name in ("make_fixtures.py", "run_gate.py") or path == Path(
                __file__
            ).resolve():
                continue
            if "fixtures/acceptance" in path.read_text():
                offenders.append(path.name)
        assert not offenders, offenders

    def test_manifest_matches_frozen_fixture_bytes(self):
        """R1-14: MANIFEST.json pins every fixture's sha256 plus the generator's own
        hash; any drift from the frozen artifact fails here."""
        import hashlib
        import json

        fixtures = DEV_FIXTURES.parent
        manifest = json.loads((fixtures / "MANIFEST.json").read_text())
        assert manifest["acceptance_seed_matrix"], "seed matrix must be recorded"
        assert len(set(manifest["acceptance_seed_matrix"])) == len(
            manifest["acceptance_seed_matrix"]
        )
        listed = set(manifest["files"])
        actual = {
            f"dev/{p.name}" for p in DEV_FIXTURES.glob("*.json")
        } | {f"acceptance/{p.name}" for p in ACCEPTANCE.glob("*.json")}
        assert listed == actual
        for rel, expected in manifest["files"].items():
            digest = hashlib.sha256((fixtures / rel).read_bytes()).hexdigest()
            assert digest == expected, f"{rel} drifted from MANIFEST.json"
        gen = hashlib.sha256((fixtures / "make_fixtures.py").read_bytes()).hexdigest()
        assert gen == manifest["generator_sha256"], "generator drifted from manifest"

    def test_acceptance_seeds_leak_nowhere(self):
        """R1-14/T6: the acceptance seed matrix must not appear in any test or sim
        source outside the manifest, the generator, and the one-shot gate driver."""
        import json

        repo = Path(__file__).resolve().parent.parent
        manifest = json.loads((repo / "sim" / "fixtures" / "MANIFEST.json").read_text())
        seeds = [str(s) for s in manifest["acceptance_seed_matrix"]]
        offenders = []
        scan = [
            *sorted((repo / "sim").rglob("*.py")),
            *sorted((repo / "tests").rglob("*.py")),
            *sorted((repo / "analysis").rglob("*.py")),
            *sorted((repo / "evolution").rglob("*.py")),
        ]
        for path in scan:
            if path.name in ("make_fixtures.py", "run_gate.py"):
                continue
            text = path.read_text()
            for seed in seeds:
                if seed in text:
                    offenders.append(f"{path}: {seed}")
        assert not offenders, offenders

    def test_gate_driver_is_one_shot_and_criteria_gated(self):
        """The gate driver refuses casual execution: env confirmation + registered
        criteria + no prior result + manifest verification (bytes inspected only)."""
        text = (ACCEPTANCE / "run_gate.py").read_text()
        assert "CONFIRM_P1_GATE" in text
        assert "GATE_RESULT.json" in text
        assert "P1_DES_GATE.json" in text
        assert "criteria_sha256" in text
        assert "driver_sha256" in text
        assert "environment_sha256" in text
        assert "source_closure_sha256" in text
        assert "honest-operator-local-marker" in text
        assert "--untracked-files=all" in text
        assert "GATE_ATTEMPT.json" in text
        assert "O_EXCL" in text
        assert "MANIFEST.json" in text
        assert not (ACCEPTANCE / "GATE_ATTEMPT.json").exists(), (
            "the one-shot gate appears to have been attempted during development"
        )
        assert not (ACCEPTANCE / "GATE_RESULT.json").exists(), (
            "the one-shot gate appears to have been executed during development"
        )

    def test_gate_dirty_source_parser_detects_tracked_and_untracked_changes(self):
        """R3-3: the clean-source preflight cannot ignore untracked execution code."""
        import importlib.util

        driver = ACCEPTANCE / "run_gate.py"
        module_spec = importlib.util.spec_from_file_location("phase2_gate_driver", driver)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        status = " M sim/engine.py\n?? sim/injected.py\n"
        assert module._parse_dirty_source_status(status) == (
            " M sim/engine.py",
            "?? sim/injected.py",
        )

    def test_gate_hashes_the_native_extension_not_the_package_shim(self):
        """R4-1: source closure pins placementscript.abi3.so, not __init__.py."""
        import importlib.machinery
        import importlib.util

        driver = ACCEPTANCE / "run_gate.py"
        module_spec = importlib.util.spec_from_file_location("phase2_gate_native", driver)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        native = module._native_artifact_path()
        assert any(
            native.name.endswith(suffix)
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
        )
        hashes = module._execution_source_hashes()
        shim, resolved_native = module._placementscript_artifacts()
        assert resolved_native == native
        assert hashes["@python/placementscript/__init__.py"] == module._sha_bytes(
            shim.read_bytes()
        )
        assert hashes[f"@native/{native.name}"] == module._sha_bytes(native.read_bytes())
        assert module._sha_bytes(native.read_bytes() + b"changed") != hashes[
            f"@native/{native.name}"
        ]
