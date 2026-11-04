"""Snapshot/reset API (TODO 2.6) and L0 analytical evaluator (TODO 2.7) tests."""

from __future__ import annotations

import json
import time

import placementscript as ps
import pytest
from sim_helpers import (
    DEV_FIXTURES,
    POLICIES,
    fanout_config,
    single_service_config,
    spread_scorer_fn,
    two_edge_cluster,
)

from sim import (
    BoundaryWeights,
    CallableScorer,
    L0Evaluator,
    PlacementScriptScorer,
    Simulation,
    Snapshot,
    SnapshotLibrary,
    export_library,
    initial_snapshot,
    load_episode_config,
    run_episode,
)

SEED = 31337


def spread():
    return CallableScorer("spread", spread_scorer_fn)


def small_config():
    return load_episode_config(DEV_FIXTURES / "small_cluster.json")


class TestSnapshotCapture:
    def test_timed_and_cycle_snapshots_are_taken(self):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        assert len(result.snapshots) >= len(config.snapshot_times_s)
        cycle_snaps = [s for s in result.snapshots if s.pending_dense_ids]
        assert cycle_snaps, "decision-point snapshots must contain pending pods"

    def test_pending_pods_are_in_arrival_order_and_unbound(self):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        for snap in result.snapshots:
            ids = list(snap.pending_dense_ids)
            assert ids == sorted(ids)
            for dense in ids:
                assert snap.ctx["pods"][dense]["current_node"] == -1

    def test_snapshot_ctx_is_dense_and_spec_shaped(self):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        snap = result.snapshots[-1]
        for i, node in enumerate(snap.ctx["nodes"]):
            assert node["id"] == i
            assert len(node["latency_to_ms"]) == len(snap.ctx["nodes"])
        for i, pod in enumerate(snap.ctx["pods"]):
            assert pod["id"] == i
        graph = snap.ctx["graph"]
        n_edges = len(graph["src"])
        assert (
            len(graph["dst"]) == len(graph["rate"]) == len(graph["payload_kb"]) == n_edges
        )

    def test_snapshot_ctx_is_directly_scorable_by_native_policy(self):
        """The exported ctx (thawed via to_dict — snapshots are deep-frozen) must
        satisfy the spec §4.1 host input domain."""
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        policy = ps.Policy((POLICIES / "energy_greedy.ps").read_text())
        snap = next(s for s in result.snapshots if s.pending_dense_ids)
        ctx = snap.to_dict()["ctx"]
        pod = ctx["pods"][snap.pending_dense_ids[0]]
        scores = policy.score_batch(pod, ctx["nodes"], ctx)
        assert len(scores) == len(ctx["nodes"])

    def test_snapshot_payload_is_deep_frozen(self):
        """Snapshots are immutable all the way down (R1-8): consumers cannot corrupt a
        shared library through aliasing."""
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        snap = result.snapshots[-1]
        with pytest.raises(TypeError):
            snap.ctx["now_ms"] = 1
        with pytest.raises((TypeError, AttributeError)):
            snap.ctx["nodes"][0]["cpu_allocated"] = 10**9
        with pytest.raises((TypeError, AttributeError)):
            snap.ctx["nodes"].append({})
        # to_dict() copies are plain and freely mutable without touching the original.
        original = snap.ctx["nodes"][0]["cpu_allocated"]
        thawed = snap.to_dict()
        thawed["ctx"]["nodes"][0]["cpu_allocated"] = original + 123
        assert snap.ctx["nodes"][0]["cpu_allocated"] == original
        assert snap.to_dict()["ctx"]["nodes"][0]["cpu_allocated"] == original

    def test_binding_pods_are_overlaid_not_pending(self):
        """Scheduler-view completeness (R1-8): a pod whose binding is in flight is
        excluded from pending and its reservation is folded into node allocations."""
        config = small_config()
        sim = Simulation(config, SEED, spread())
        sim.start()
        # Find an instant with an in-flight binding by stepping through events.
        found = False
        while sim.env.peek() < config.horizon_s:
            sim.env.step()
            if sim.scheduler.cache.assumed:
                found = True
                break
        assert found, "no in-flight binding ever observed"
        snap = sim.snapshot_now()
        ctx = snap.to_dict()["ctx"]
        assumed = dict(sim.scheduler.cache.assumed)
        assert snap.binding_dense_ids, "binding pods must be listed"
        for dense in snap.binding_dense_ids:
            assert dense not in snap.pending_dense_ids
            assert ctx["pods"][dense]["current_node"] >= 0
        # The overlaid allocation equals truth plus the assumed reservations, exactly.
        expected = [n.cpu_allocated for n in sim.cluster.nodes]
        for node_id, cpu, _mem in assumed.values():
            expected[node_id] += cpu
        for node in ctx["nodes"]:
            assert node["cpu_allocated"] == expected[node["id"]]

    def test_snapshot_semantic_validation_rejects_forged_ids_and_time(self):
        """R3-5: a digest-valid but semantically impossible snapshot is refused."""
        pristine = initial_snapshot(small_config()).to_dict()

        def rejects(mutate, message):
            payload = json.loads(json.dumps(pristine))
            mutate(payload)
            with pytest.raises(ValueError, match=message):
                Snapshot.from_dict(payload)

        rejects(
            lambda p: p.__setitem__("binding_dense_ids", [p["pending_dense_ids"][0]]),
            "disjoint",
        )
        rejects(
            lambda p: p.__setitem__("pending_dense_ids", [len(p["ctx"]["pods"])]),
            "outside",
        )
        rejects(
            lambda p: p["ctx"].__setitem__("now_ms", 1),
            "match taken_at_s",
        )
        rejects(
            lambda p: p["pod_phases"].__setitem__(0, "ready"),
            "PENDING lifecycle",
        )
        rejects(
            lambda p: p["ctx"]["nodes"][0].__setitem__("power_idle_w", 1),
            "must be a float",
        )

    def test_terminating_pod_does_not_supply_l0_queueing_capacity(self):
        """R3-2: placed lifecycle state affects allocation, but READY alone serves."""
        config = single_service_config(two_edge_cluster(), rate=1.0, horizon_s=5.0)
        scorer = CallableScorer("flat", lambda pod, node, ctx: 1.0)
        sim = Simulation(config, SEED, scorer)
        sim.start()
        while not sim.app.services[0].replicas:
            sim.env.step()
        pod = next(iter(sim.app.services[0].replicas.values())).pod
        sim.app.replica_leaving(pod)
        pod.phase = "terminating"
        snap = sim.snapshot_now()
        assert snap.pod_phases[pod.uid] == "terminating"
        out = L0Evaluator(
            SnapshotLibrary(config, (snap,), {}), BoundaryWeights()
        ).evaluate(scorer)
        assert out.valid and not out.feasible
        assert any("no_replica" in violation for violation in out.violations)

    @pytest.mark.parametrize(
        ("mode", "expected_payload"), (("blocking", 3.0), ("forward", 1.0))
    )
    def test_call_mode_payload_crosses_native_boundary_exactly(
        self, mode, expected_payload
    ):
        """R4-2/R6: wire reals are floats; forward calls advertise request only."""
        from sim import (
            ArrivalSpec,
            CallEdgeSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            SchedulerParams,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
            det,
        )

        node_class = NodeClassSpec("integer-input", 1, 2, image_pull_mib_per_s=100)
        cluster = ClusterSpec(
            (node_class,),
            (NodeSpec(0, "integer-input", 2, 1000, 1000),),
            ((0,),),
        )
        services = tuple(
            ServiceSpec(
                sid,
                f"s{sid}",
                0,
                100,
                10,
                det(0),
                100,
                image_mb=0,
            )
            for sid in range(2)
        )
        template = TemplateSpec(
            0,
            "t",
            CallSpec(
                0,
                (CallEdgeSpec(CallSpec(1), request_kb=1, response_kb=2),),
                mode=mode,
            ),
            ArrivalSpec("deterministic", 1, start_s=0.5),
        )
        config = EpisodeConfig(
            cluster,
            WorkloadSpec(services, (template,)),
            2,
            arrival_cutoff_s=1.0,
            scheduler=SchedulerParams(bind_latency=det(0)),
        )
        py_scorer = CallableScorer("flat", lambda pod, node, ctx: 1.0)
        sim = Simulation(config, SEED, py_scorer)
        sim.start()
        policy = ps.Policy((POLICIES / "least_allocated.ps").read_text())
        native = PlacementScriptScorer(policy)
        for at, expected_rate in (
            (0.1, 0.0),
            (0.5, 1.0),
            (0.75, 1.0),
            (1.0, 0.0),
            (1.1, 0.0),
        ):
            sim.advance(at)
            snap = sim.snapshot_now()
            ctx = snap.to_dict()["ctx"]
            assert ctx["graph"]["rate"] == [expected_rate]
            assert ctx["graph"]["payload_kb"] == [expected_payload]
            assert type(ctx["nodes"][0]["power_idle_w"]) is float
            assert type(ctx["nodes"][0]["latency_to_ms"][0]) is float
            assert type(ctx["graph"]["rate"][0]) is float
            assert type(ctx["graph"]["payload_kb"][0]) is float
            assert len(native.score_batch(ctx["pods"][0], ctx["nodes"], ctx)) == 1
            # The native proposer consumes the same graph contract used by the
            # rebalancer path; this is a validation probe, not an actuation request.
            policy.propose_moves(
                {"nodes": ctx["nodes"], "pods": ctx["pods"]},
                ctx["graph"],
                0,
            )


class TestLibraryRoundTrip:
    def test_save_load_digest_stable(self, tmp_path):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        library = export_library(config, result.snapshots, meta={"seed": SEED})
        path = tmp_path / "lib.json"
        digest = library.save(path)
        loaded = SnapshotLibrary.load(path)
        assert loaded.digest() == library.digest() == digest
        assert loaded.meta == {"seed": SEED}
        assert len(loaded.snapshots) == len(library.snapshots)
        assert loaded.snapshots[0].to_dict() == library.snapshots[0].to_dict()

    def test_tampered_library_rejected(self, tmp_path):
        """The whole payload is digest-protected (R1-8): config, snapshot contents, and
        metadata edits are all rejected, not just config changes."""
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        library = export_library(config, result.snapshots, meta={"root_seed": SEED})
        path = tmp_path / "lib.json"
        library.save(path)
        pristine = path.read_text()

        def tamper(mutate):
            blob = json.loads(pristine)
            mutate(blob["payload"])
            path.write_text(json.dumps(blob))
            with pytest.raises(ValueError, match="digest mismatch|hash mismatch"):
                SnapshotLibrary.load(path)

        tamper(lambda p: p["config"].__setitem__("horizon_s", 999.0))
        tamper(
            lambda p: p["snapshots"][0]["ctx"]["nodes"][0].__setitem__(
                "cpu_allocated", 10**8
            )
        )
        tamper(lambda p: p["meta"].__setitem__("root_seed", SEED + 1))
        tamper(lambda p: p["snapshots"].pop())

    def test_external_digest_pins_the_frozen_artifact(self, tmp_path):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        library = export_library(config, result.snapshots, meta={})
        path = tmp_path / "lib.json"
        digest = library.save(path)
        SnapshotLibrary.load(path, expected_digest=digest)  # exact artifact: fine
        with pytest.raises(ValueError, match="digest mismatch"):
            SnapshotLibrary.load(path, expected_digest="0" * 64)

    def test_empty_export_refused(self):
        with pytest.raises(ValueError, match="empty"):
            export_library(small_config(), [], meta={})

    def test_public_library_collections_are_deep_frozen(self):
        from dataclasses import FrozenInstanceError

        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        library = export_library(
            config, result.snapshots, meta={"seeds": [SEED], "nested": {"x": 1}}
        )
        with pytest.raises((AttributeError, TypeError)):
            library.snapshots.append(result.snapshots[0])
        with pytest.raises(TypeError):
            library.meta["nested"]["x"] = 2
        with pytest.raises(FrozenInstanceError):
            library.snapshots = ()


class TestReplayRestore:
    def test_advance_to_t_reproduces_full_run_state_at_t(self):
        """The 'reset/restore' contract: replaying to t yields exactly the state the
        full run had at t (snapshot equality is the observable witness)."""
        config = small_config()
        t = config.snapshot_times_s[1]
        full = run_episode(config, seed=SEED, scorer=spread())
        full_snap = next(s for s in full.snapshots if s.taken_at_s == t)

        partial = Simulation(config, SEED, spread())
        partial.start()
        partial.advance(t)
        restored = partial.snapshot_now()
        assert restored.to_dict() == full_snap.to_dict()

    def test_two_replays_agree_at_intermediate_time(self):
        config = fanout_config(horizon_s=30.0)
        snaps = []
        for _ in range(2):
            sim = Simulation(config, SEED, spread())
            sim.start()
            sim.advance(13.7)
            snaps.append(sim.snapshot_now().to_dict())
        assert snaps[0] == snaps[1]

    def test_simulation_runs_exactly_once(self):
        config = fanout_config(horizon_s=5.0)
        sim = Simulation(config, SEED, spread())
        sim.run()
        with pytest.raises(RuntimeError, match="exactly once"):
            sim.run()


class TestL0Evaluator:
    def _library(self):
        config = small_config()
        result = run_episode(config, seed=SEED, scorer=spread())
        return export_library(
            config, [initial_snapshot(config), *result.snapshots], meta={"seed": SEED}
        )

    def test_crash_filter_invalidates_broken_policies(self):
        library = self._library()
        evaluator = L0Evaluator(library, BoundaryWeights())

        def crash(pod, node, ctx):
            raise RuntimeError("boom")

        out = evaluator.evaluate(CallableScorer("crash", crash))
        assert out.valid is False
        assert "boom" in out.error
        assert out.mean_normalized_energy is None

        trap = ps.Policy(
            'POLICY_NAME: str = "t"\nPOLICY_VERSION: str = "1"\n'
            "def score(pod: Pod, node: NodeState, ctx: ClusterCtx) -> float:\n"
            "    return 1.0 / (float(node.cpu_allocated) * 0.0)\n"
            "def propose_moves(state: ClusterState, graph: CallGraphCtx, "
            "budget: int) -> list[Move]:\n"
            "    out: list[Move] = []\n    return out\n"
        )
        out = evaluator.evaluate(PlacementScriptScorer(trap))
        assert out.valid is False
        assert "TrapError" in out.error

    def test_deterministic_and_policy_discriminating(self):
        library = self._library()
        evaluator = L0Evaluator(library, BoundaryWeights())
        results = {}
        for name in ("least_allocated", "energy_greedy", "latency_greedy"):
            scorer = PlacementScriptScorer(
                ps.Policy((POLICIES / f"{name}.ps").read_text())
            )
            a = evaluator.evaluate(scorer)
            b = evaluator.evaluate(scorer)
            assert a.valid and b.valid
            assert a.mean_normalized_energy == b.mean_normalized_energy  # bitwise
            assert [r.decisions for r in a.per_snapshot] == [
                r.decisions for r in b.per_snapshot
            ]
            results[name] = a.mean_normalized_energy
        # Packing beats spreading on modeled energy; latency-greedy differs from both.
        assert results["energy_greedy"] < results["least_allocated"]
        assert len(set(results.values())) >= 2

    def test_normalized_energy_uses_policy_independent_idle_floor(self):
        from sim.energy import idle_floor_j

        library = self._library()
        evaluator = L0Evaluator(library, BoundaryWeights())
        config = library.config
        floor = idle_floor_j(
            config.cluster, config.energy, config.horizon_s, BoundaryWeights()
        )
        scorer = PlacementScriptScorer(
            ps.Policy((POLICIES / "energy_greedy.ps").read_text())
        )
        out = evaluator.evaluate(scorer)
        for r in out.per_snapshot:
            assert r.normalized_energy == pytest.approx(
                r.closed_boundary_j / floor, rel=1e-12
            )

    def test_peak_rate_respects_template_activity_windows(self):
        """R2-6: a template wholly outside the cutoff contributes no peak load."""
        from sim import (
            ArrivalSpec,
            CallSpec,
            ClusterSpec,
            EpisodeConfig,
            NodeClassSpec,
            NodeSpec,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
            det,
        )

        cls = NodeClassSpec("n", 1.0, 2.0)
        cluster = ClusterSpec(
            (cls,), (NodeSpec(0, "n", 2, 1000, 1000),), ((0.0,),)
        )
        service = ServiceSpec(
            0, "s", 0, 100, 10, det(1.0), 0,
            concurrency=1, image_mb=0,
        )
        inactive = TemplateSpec(
            0,
            "inactive",
            CallSpec(0),
            ArrivalSpec("deterministic", 2.0, start_s=20.0),
        )
        config = EpisodeConfig(
            cluster,
            WorkloadSpec((service,), (inactive,)),
            horizon_s=30.0,
            arrival_cutoff_s=10.0,
        )
        library = SnapshotLibrary(config, (initial_snapshot(config),), {})
        out = L0Evaluator(library, BoundaryWeights()).evaluate(
            CallableScorer("flat", lambda pod, node, ctx: 1.0)
        )
        assert out.valid and out.feasible
        assert not any("saturated" in v for v in out.violations)

        # Once two windows actually overlap, their rates must be summed at the peak.
        a = TemplateSpec(
            0, "a", CallSpec(0), ArrivalSpec("deterministic", 0.6, start_s=0.0)
        )
        b = TemplateSpec(
            1, "b", CallSpec(0), ArrivalSpec("deterministic", 0.6, start_s=5.0)
        )
        overlap = EpisodeConfig(
            cluster,
            WorkloadSpec((service,), (a, b)),
            horizon_s=20.0,
            arrival_cutoff_s=10.0,
        )
        overlap_out = L0Evaluator(
            SnapshotLibrary(overlap, (initial_snapshot(overlap),), {}),
            BoundaryWeights(),
        ).evaluate(CallableScorer("flat", lambda pod, node, ctx: 1.0))
        assert any("saturated" in v for v in overlap_out.violations)

    def test_l0_honors_quantize_scores_flag(self):
        """R1-9 regression: L0 runs the same Normalize step as the configured emulator.
        Raw scores (0.0, 99.6, 100.0) quantize (affine + round) to (0, 100, 100) ->
        tie-break picks node 1; without quantization node 2 wins. L0 must match the
        emulator in both modes."""
        from dataclasses import replace

        from sim import (
            ArrivalSpec,
            CallSpec,
            ClusterSpec,
            NodeClassSpec,
            NodeSpec,
            ServiceSpec,
            TemplateSpec,
            WorkloadSpec,
        )
        from sim.spec import EpisodeConfig, expo

        cls = NodeClassSpec(name="n", power_idle_w=1.0, power_max_w=2.0)
        cluster = ClusterSpec(
            classes=(cls,),
            nodes=(
                NodeSpec(0, "n", 2, 4000, 8192),
                NodeSpec(1, "n", 2, 4000, 8192),
                NodeSpec(2, "n", 2, 4000, 8192),
            ),
            latency_ms=((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
        )
        svc = ServiceSpec(
            id=0, name="s", app=0, cpu_request=100, mem_request=64,
            service_time=expo(0.01), cpu_demand=100, image_mb=1,
        )
        template = TemplateSpec(
            id=0, name="t", root=CallSpec(service=0), arrival=ArrivalSpec("poisson", 1.0)
        )
        config = EpisodeConfig(
            cluster=cluster,
            workload=WorkloadSpec(services=(svc,), templates=(template,)),
            horizon_s=10.0,
        )

        def graded(pod, node, ctx):
            del pod, ctx
            return {0: 0.0, 1: 99.6, 2: 100.0}[node["id"]]

        scorer = CallableScorer("graded", graded)
        for quantize, expected_node in ((True, 1), (False, 2)):
            cfg = replace(config, scheduler=replace(config.scheduler, quantize_scores=quantize))
            snap = initial_snapshot(cfg)
            library = SnapshotLibrary(config=cfg, snapshots=[snap], meta={})
            out = L0Evaluator(library, BoundaryWeights()).evaluate(scorer)
            assert out.valid
            assert out.per_snapshot[0].decisions == [(0, expected_node)]
            # And the emulator agrees under the same config.
            result = run_episode(cfg, seed=SEED, scorer=scorer)
            first = next(d for d in result.decision_log if d["event"] == "reserve")
            assert first["node"] == expected_node

    def test_host_errors_are_never_charged_to_the_candidate(self):
        """R1-12: a scorer that raises the *host's* error type must crash the L0
        evaluation, not be absorbed as an invalid candidate."""

        from sim.runtime import HostError

        class HostFault:
            name = "py:hostfault"

            def score_batch(self, pod, nodes, ctx):
                raise HostError("simulated host-side failure")

        library = self._library()
        evaluator = L0Evaluator(library, BoundaryWeights())
        with pytest.raises(HostError):
            evaluator.evaluate(HostFault())

    def test_class_name_cannot_spoof_a_host_error(self):
        """R2-8: only the explicit host type crosses the policy boundary."""

        class HostError(Exception):
            pass

        class Spoof:
            name = "py:spoof"

            def score_batch(self, pod, nodes, ctx):
                raise HostError("candidate exception with a spoofed class name")

        out = L0Evaluator(self._library(), BoundaryWeights()).evaluate(Spoof())
        assert not out.valid
        assert "spoofed class name" in out.error

    def test_benchmark_is_of_order_one_ms_per_candidate(self):
        """The plan's L0 cost target (§3.3: ~1 ms). The recorded benchmark is
        sim/benchmarks/l0_bench.py; this is a smoke bound with generous headroom so CI
        machines do not flap."""
        library = self._library()
        evaluator = L0Evaluator(library, BoundaryWeights())
        scorer = PlacementScriptScorer(
            ps.Policy((POLICIES / "least_allocated.ps").read_text())
        )
        evaluator.evaluate(scorer)  # warm-up
        n = 30
        t0 = time.perf_counter()
        for _ in range(n):
            evaluator.evaluate(scorer)
        per_candidate_ms = (time.perf_counter() - t0) * 1000.0 / n
        print(f"\nL0: {per_candidate_ms:.3f} ms/candidate over {len(library.snapshots)} snapshots")
        assert per_candidate_ms < 10.0

    def test_synthetic_benchmark_honors_odd_pending_count(self):
        """R3-6: --pending is exact for odd as well as even values."""
        from sim.benchmarks.l0_bench import synthetic_library

        library = synthetic_library(nodes=3, snapshots=2, pending=13)
        assert [len(snapshot.pending_dense_ids) for snapshot in library.snapshots] == [
            13,
            13,
        ]
