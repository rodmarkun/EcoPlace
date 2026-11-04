"""Bundle assembly tests (TODO 3.2/3.3): capacity-mapping rules, parametric fit,
joint-tuple discipline (no impossible devices), split invariants, Phase 2 bridge."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sim.energy.bundles import (
    FAN_DEFAULT_R,
    PARAMETRIC,
    PIECEWISE,
    build_development_bundles,
    build_held_out_bundles,
    bundle_from_registry,
    default_capacity_mapping,
    fan_power_w,
    fit_fan_exponent,
    machine_tuple_from_registry,
    validate_bundle,
    validate_split,
)
from sim.energy.capacity import LIMITATIONS, CapacityMapping
from sim.energy.sources import load_pinned_sources


@pytest.fixture(scope="module")
def registry():
    return load_pinned_sources()


@pytest.fixture(scope="module")
def mapping(registry):
    return default_capacity_mapping(registry)


class TestCapacityMapping:
    def test_reference_machine_maps_to_its_declared_anchor(self, registry, mapping):
        anchor = mapping.anchor_for("specpower-ssj2008")
        ref = registry.specpower[anchor.reference_machine_id]
        assert mapping.capacity_millicores(ref) == anchor.reference_capacity_millicores

    def test_throughput_anchoring_is_linear_in_peak_ops(self, registry, mapping):
        anchor = mapping.anchor_for("specpower-ssj2008")
        ref = registry.specpower[anchor.reference_machine_id]
        other = registry.specpower["specpower-supermicro-as1115sv-2023"]
        expected = round(
            anchor.reference_capacity_millicores * other.peak_ssj_ops / ref.peak_ssj_ops
        )
        assert mapping.capacity_millicores(other) == expected
        assert mapping.rule_used(other) == "throughput-anchored[specpower-ssj2008]"

    def test_arm_boards_anchor_within_the_hpl_domain(self, registry, mapping):
        """R1-1: the cited paper publishes HPL GFLOPS for both boards; capacities
        scale by the published ratio inside the HPL domain, never against ssj_ops."""
        pi = registry.arm_sbc["armsbc-rpi3b-cloutier2016"]
        jetson = registry.arm_sbc["armsbc-jetsontx1-cloutier2016"]
        assert pi.benchmark_domain == "hpl-gflops-cpu-dp"
        assert mapping.capacity_millicores(pi) == 4000  # the declared HPL anchor
        assert mapping.capacity_millicores(jetson) == round(4000 * 16.0 / 3.7)
        assert mapping.rule_used(jetson) == "throughput-anchored[hpl-gflops-cpu-dp]"

    def test_board_without_published_throughput_uses_core_count_fallback(self, registry, mapping):
        kaup = registry.arm_sbc["armsbc-rpib-kaup2014"]
        assert kaup.peak_throughput_ops is None
        assert mapping.capacity_millicores(kaup) == 1 * 1000
        assert mapping.rule_used(kaup) == "core-count-fallback"

    def test_throughput_in_unanchored_domain_is_rejected_not_mapped(self, registry):
        """Cross-domain anchoring must fail loudly (R1-1: never compare HPL GFLOPS
        with ssj_ops)."""
        spec_only = CapacityMapping.anchored_on(
            registry.specpower["specpower-fujitsu-rx100s7-2011"], 4000
        )
        jetson = registry.arm_sbc["armsbc-jetsontx1-cloutier2016"]
        with pytest.raises(ValueError, match="no capacity anchor.*hpl-gflops"):
            spec_only.capacity_millicores(jetson)

    def test_limitations_are_enumerated_for_threats_to_validity(self):
        assert len(LIMITATIONS) >= 5
        assert any("SPECpower" in item for item in LIMITATIONS)
        assert any("fallback" in item for item in LIMITATIONS)

    def test_mapping_rejects_nonpositive_anchor(self, registry):
        ref = registry.specpower["specpower-fujitsu-rx100s7-2011"]
        with pytest.raises(ValueError):
            CapacityMapping.anchored_on(ref, 0)

    def test_specpower_curve_coordinate_is_measured_throughput_fraction(self, registry):
        """R1-1: the joint curve consumes ssj_ops/peak, not the nominal target label."""
        m = registry.specpower["specpower-fujitsu-rx100s7-2011"]
        pts = m.joint_curve_points()
        assert pts[0] == (0.0, m.active_idle_w)
        for (u, _), row in zip(pts[1:], m.load_points, strict=True):
            assert u == row.ssj_ops / m.peak_ssj_ops
        assert pts[-1][0] == 1.0
        # the pinned set genuinely differs between the two coordinates
        assert any(
            abs(row.target_load_frac - row.ssj_ops / m.peak_ssj_ops) > 0.005
            for row in m.load_points
        )


class TestParametricFit:
    def test_fit_matches_endpoints_exactly(self, registry):
        m = registry.specpower["specpower-ibm-x3550m3-2011"]
        pts = m.joint_curve_points()
        r, resid = fit_fan_exponent(pts)
        assert fan_power_w(0.0, pts[0][1], pts[-1][1], r) == pts[0][1]
        assert fan_power_w(1.0, pts[0][1], pts[-1][1], r) == pts[-1][1]
        assert resid >= 0.0

    def test_fit_is_deterministic(self, registry):
        m = registry.specpower["specpower-asus-esc4000e11-2023"]
        assert fit_fan_exponent(m.joint_curve_points()) == fit_fan_exponent(
            m.joint_curve_points()
        )

    def test_two_point_machines_use_published_fan_exponent(self, registry, mapping):
        t = machine_tuple_from_registry(
            registry, mapping, node_class_name="far-edge",
            machine_id="armsbc-jetsontx1-cloutier2016", curve_family=PARAMETRIC,
        )
        assert t.fit_exponent_r == FAN_DEFAULT_R

    def test_fit_beats_default_exponent_on_rich_curves(self, registry):
        """The deterministic fit must be at least as good (SSE) as the fixed default."""
        for mid in ("specpower-asus-esc4000e11-2023", "specpower-h3c-r6900g6-2023"):
            pts = registry.specpower[mid].joint_curve_points()
            r, _ = fit_fan_exponent(pts)

            def sse(rr, pts=pts):
                return sum((fan_power_w(u, pts[0][1], pts[-1][1], rr) - w) ** 2 for u, w in pts)

            assert sse(r) <= sse(FAN_DEFAULT_R) + 1e-9

    def test_fit_exponent_constrained_to_monotone_interval(self, registry):
        """R1-2: unconstrained fits on the 4-socket machines chose r < 1 and produced
        curves dipping below idle; the fit interval is now [1, 2]."""
        for mid in ("specpower-asus-esc4000e11-2023", "specpower-h3c-r6900g6-2023"):
            r, _ = fit_fan_exponent(registry.specpower[mid].joint_curve_points())
            assert 1.0 <= r <= 2.0

    def test_every_parametric_curve_is_nondecreasing_and_at_least_idle(self, registry, mapping):
        """R1-2: a policy must never be rewarded for adding load."""
        ho = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)
        for b in ho:
            for m in b.machines:
                idle = m.power_curve_points[0][1]
                powers = [w for _, w in m.power_curve_points]
                assert all(w >= idle for w in powers)
                assert all(b_ >= a_ for a_, b_ in zip(powers, powers[1:], strict=False))

    def test_out_of_interval_exponent_rejected(self, registry):
        from sim.energy.bundles import parametric_grid

        pts = registry.specpower["specpower-ibm-x3550m3-2011"].joint_curve_points()
        with pytest.raises(ValueError, match="monotone interval"):
            parametric_grid(pts, 0.72)


class TestJointTupleDiscipline:
    def test_tuple_rederivable_from_source(self, registry, mapping):
        for bundle in build_development_bundles(registry, mapping):
            validate_bundle(bundle, registry, mapping)  # must not raise

    def test_frankenstein_tuple_rejected(self, registry, mapping):
        """Power curve from one machine + capacity from another must fail validation
        (the 'no impossible devices' probe)."""
        [b1, b2, _] = build_development_bundles(registry, mapping)
        fog1 = next(m for m in b1.machines if m.node_class_name == "fog-server")
        fog2 = next(m for m in b2.machines if m.node_class_name == "fog-server")
        franken = dataclasses.replace(fog1, capacity_millicores=fog2.capacity_millicores)
        broken = dataclasses.replace(
            b1, machines=tuple(franken if m is fog1 else m for m in b1.machines)
        )
        with pytest.raises(ValueError, match="joint-tuple discipline"):
            validate_bundle(broken, registry, mapping)

    def test_tampered_curve_rejected(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        fog = next(m for m in b1.machines if m.node_class_name == "fog-server")
        pts = list(fog.power_curve_points)
        pts[3] = (pts[3][0], pts[3][1] * 0.9)
        tampered = dataclasses.replace(fog, power_curve_points=tuple(pts))
        broken = dataclasses.replace(
            b1, machines=tuple(tampered if m is fog else m for m in b1.machines)
        )
        with pytest.raises(ValueError, match="joint-tuple discipline"):
            validate_bundle(broken, registry, mapping)

    def test_coefficient_divergence_rejected(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        broken = dataclasses.replace(b1, cloud_j_per_vcpu_hour=b1.cloud_j_per_vcpu_hour * 1.01)
        with pytest.raises(ValueError, match="swept range|diverge|seal mismatch"):
            validate_bundle(broken, registry, mapping)

    def test_forged_tuple_fails_at_consumption_without_registry(self, registry, mapping):
        """R1-3: a field-swapped tuple must fail at the point of use (seal check),
        even when the caller never runs validate_bundle."""
        [b1, b2, _] = build_development_bundles(registry, mapping)
        fog1 = next(m for m in b1.machines if m.node_class_name == "fog-server")
        fog2 = next(m for m in b2.machines if m.node_class_name == "fog-server")
        franken = dataclasses.replace(fog1, capacity_millicores=fog2.capacity_millicores)
        with pytest.raises(ValueError, match="seal mismatch"):
            franken.node_class_spec()

    def test_forged_bundle_fails_at_consumption_without_registry(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        broken = dataclasses.replace(
            b1, network_j_per_byte=b1.network_j_per_byte * 1.5
        )
        with pytest.raises(ValueError, match="seal mismatch"):
            broken.energy_coefficients()
        with pytest.raises(ValueError, match="seal mismatch"):
            broken.node_class_specs()

    def test_sealless_construction_rejected(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        fog = next(m for m in b1.machines if m.node_class_name == "fog-server")
        with pytest.raises(ValueError, match="seal"):
            dataclasses.replace(fog, seal="")

    def test_mixed_curve_families_rejected_within_bundle(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        fog_id = next(
            m.machine_id for m in b1.machines if m.node_class_name == "fog-server"
        )
        parametric_fog = machine_tuple_from_registry(
            registry, mapping, node_class_name="fog-server",
            machine_id=fog_id, curve_family=PARAMETRIC,
        )
        with pytest.raises(ValueError, match="mixed curve families"):
            dataclasses.replace(
                b1,
                machines=tuple(
                    parametric_fog if m.node_class_name == "fog-server" else m
                    for m in b1.machines
                ),
            )


class TestSplit:
    def test_draft_split_is_valid(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        ho = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)
        validate_split(dev, ho)

    def test_split_registration_payload_tampering_is_rejected(self, tmp_path):
        """R7-1: role ids/rosters are protected by a load-bearing content hash."""
        from sim.energy import registration as registration_module

        raw = json.loads(registration_module._REGISTRATION_PATH.read_text())
        raw["split"]["payload"]["roles"][0]["node_class_roster"].append("invented")
        tampered = tmp_path / "energy_analysis_registration.json"
        tampered.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="split registration seal mismatch"):
            registration_module._load_energy_analysis_registration(tampered)

    def test_held_out_requires_acknowledgement(self, registry):
        with pytest.raises(PermissionError, match="holdout_gate"):
            build_held_out_bundles(registry)

    def test_shared_machine_violates_split(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        with pytest.raises(
            ValueError,
            match=(
                "bundle ids must be unique|canonical registered set|"
                "shared machines|shared curve families"
            ),
        ):
            validate_split(dev, dev)

    def test_empty_split_is_rejected(self):
        """R5-1 exact reviewer probe: an empty split must never certify separation."""
        with pytest.raises(ValueError, match="non-empty"):
            validate_split((), ())

    def test_capacity_tampered_held_out_bundle_is_rejected(self, registry, mapping):
        """R5-1 exact reviewer probe: split validation consumes verified manifests."""
        dev = build_development_bundles(registry, mapping)
        held_out = build_held_out_bundles(
            registry, mapping, acknowledged_one_shot=True
        )
        machine = held_out[0].machines[0]
        tampered_machine = dataclasses.replace(
            machine, capacity_millicores=machine.capacity_millicores + 1
        )
        tampered_bundle = dataclasses.replace(
            held_out[0],
            machines=(tampered_machine, *held_out[0].machines[1:]),
        )
        with pytest.raises(ValueError, match="seal mismatch"):
            tampered_bundle.verify_seal()
        with pytest.raises(ValueError, match="seal mismatch"):
            validate_split(dev, (tampered_bundle, *held_out[1:]))

    def test_duplicate_bundle_ids_are_not_a_complete_set(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        held_out = build_held_out_bundles(
            registry, mapping, acknowledged_one_shot=True
        )
        with pytest.raises(ValueError, match="bundle ids must be unique"):
            validate_split((dev[0], dev[0]), held_out)

    def test_canonical_bundle_subsets_are_rejected(self, registry, mapping):
        """R5-1/R6 exact probe: completeness comes from the internal registry,
        never from the first caller-supplied bundle."""
        dev = build_development_bundles(registry, mapping)
        held_out = build_held_out_bundles(
            registry, mapping, acknowledged_one_shot=True
        )
        with pytest.raises(ValueError, match="canonical registered set"):
            validate_split(dev[:1], held_out[:1])

    def test_public_factory_fog_only_pair_is_rejected(self, registry, mapping):
        """R5-1/R6 exact probe: two valid, sealed public-factory bundles cannot
        redefine the canonical complete node-class roster."""
        dev = build_development_bundles(registry, mapping)[0]
        held_out = build_held_out_bundles(
            registry, mapping, acknowledged_one_shot=True
        )[0]

        def fog_only(bundle):
            fog = next(
                machine
                for machine in bundle.machines
                if machine.node_class_name == "fog-server"
            )
            return bundle_from_registry(
                registry,
                mapping,
                bundle_id=bundle.bundle_id,
                role=bundle.role,
                curve_family=bundle.curve_family,
                machines=(
                    (
                        fog.node_class_name,
                        fog.machine_id,
                        fog.transition_entry_id,
                    ),
                ),
                cloud_entry_id=bundle.cloud_entry_id,
                network_entry_id=bundle.network_entry_id,
            )

        with pytest.raises(ValueError, match="canonical registered set|complete roster"):
            validate_split((fog_only(dev),), (fog_only(held_out),))

    def test_builder_monkeypatch_cannot_redefine_registered_split(
        self, registry, mapping, monkeypatch
    ):
        """R7-1: sealed role identities/rosters, not live builders, are authority."""
        import sim.energy.bundles as bundles_module

        development = build_development_bundles(registry, mapping)
        held_out = build_held_out_bundles(
            registry, mapping, acknowledged_one_shot=True
        )
        monkeypatch.setattr(
            bundles_module,
            "build_development_bundles",
            lambda *_args, **_kwargs: development[:1],
        )
        monkeypatch.setattr(
            bundles_module,
            "build_held_out_bundles",
            lambda *_args, **_kwargs: held_out[:1],
        )
        with pytest.raises(ValueError, match="canonical registered set"):
            validate_split(development[:1], held_out[:1])
        validate_split(development, held_out)

    def test_split_families_differ(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        ho = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)
        assert {b.curve_family for b in dev} == {PIECEWISE}
        assert {b.curve_family for b in ho} == {PARAMETRIC}

    def test_split_alloc_entries_differ(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        ho = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)
        assert {b.cloud_entry_id for b in dev}.isdisjoint({b.cloud_entry_id for b in ho})
        assert {b.network_entry_id for b in dev}.isdisjoint(
            {b.network_entry_id for b in ho}
        )


class TestRegistryBackedFactory:
    """R2-5: the public bundle factory accepts registry entry ids only — there is
    no public path that seals caller-supplied coefficient values."""

    def test_public_factory_has_no_value_parameters(self):
        import inspect

        from sim.energy.bundles import bundle_from_registry

        params = inspect.signature(bundle_from_registry).parameters
        for value_param in (
            "cloud_j_per_vcpu_hour",
            "cloud_swept_j_per_vcpu_hour",
            "network_j_per_byte",
            "network_swept_j_per_byte",
        ):
            assert value_param not in params

    def test_raw_assembler_is_private(self):
        import sim.energy
        import sim.energy.bundles as bundles_mod

        assert "assemble_bundle" not in sim.energy.__all__
        assert not hasattr(sim.energy, "assemble_bundle")
        assert not hasattr(bundles_mod, "assemble_bundle")  # renamed _assemble_bundle

    def test_factory_rejects_unknown_or_unverified_entries(self, registry, mapping):
        from sim.energy.bundles import PIECEWISE, bundle_from_registry

        with pytest.raises(ValueError, match="VERIFIED"):
            bundle_from_registry(
                registry, mapping, bundle_id="x", role="development",
                curve_family=PIECEWISE,
                machines=(("fog-server", "no-such-machine", None),),
                cloud_entry_id="ccf-aws-cpu", network_entry_id="ccf-network",
            )
        with pytest.raises(ValueError, match="VERIFIED"):
            bundle_from_registry(
                registry, mapping, bundle_id="x", role="development",
                curve_family=PIECEWISE,
                machines=(("fog-server", "specpower-fujitsu-rx100s7-2011", None),),
                cloud_entry_id="no-such-alloc", network_entry_id="ccf-network",
            )

    def test_factory_output_matches_registry_and_validates(self, registry, mapping):
        from sim.energy.bundles import PIECEWISE, bundle_from_registry

        b = bundle_from_registry(
            registry, mapping, bundle_id="probe-1", role="development",
            curve_family=PIECEWISE,
            machines=(("fog-server", "specpower-fujitsu-rx100s7-2011", None),),
            cloud_entry_id="ccf-aws-cpu", network_entry_id="ccf-network",
        )
        validate_bundle(b, registry, mapping)
        cloud = registry.cloud_allocation["ccf-aws-cpu"]
        assert b.cloud_j_per_vcpu_hour == cloud.point_j_per_vcpu_hour
        assert {r.entry_id for r in b.source_records} == {
            "specpower-fujitsu-rx100s7-2011", "ccf-aws-cpu", "ccf-network",
        }

    def test_node_class_binding_is_the_sealed_capacity_bridge(self, registry, mapping):
        """R2-5 adversarial regression: capacity previously had to be read straight
        off the dataclass; the binding bridge verifies the seal first, so a
        capacity-swapped tuple fails at consumption."""
        [b1, b2, _] = build_development_bundles(registry, mapping)
        bindings = b1.node_class_bindings()
        for m in b1.machines:
            assert bindings[m.node_class_name].capacity_millicores == m.capacity_millicores
            assert bindings[m.node_class_name].spec.name == m.node_class_name
        fog1 = next(m for m in b1.machines if m.node_class_name == "fog-server")
        fog2 = next(m for m in b2.machines if m.node_class_name == "fog-server")
        franken = dataclasses.replace(fog1, capacity_millicores=fog2.capacity_millicores)
        with pytest.raises(ValueError, match="seal mismatch"):
            franken.node_class_binding()

    def test_binding_not_assemblable_from_loose_values(self, registry, mapping):
        """R3-2: NodeClassBinding is tied to its sealed tuple — a binding claiming
        a different capacity or spec than the tuple's own derivation is rejected."""
        from sim.energy.bundles import NodeClassBinding

        [b1, _, _] = build_development_bundles(registry, mapping)
        fog = next(m for m in b1.machines if m.node_class_name == "fog-server")
        good = fog.node_class_binding()
        with pytest.raises(ValueError, match="diverges from its sealed machine tuple"):
            NodeClassBinding(
                spec=good.spec, capacity_millicores=999_999, machine=fog
            )

    def test_raw_tuple_constructor_not_public(self):
        """R3-2: the raw source-object tuple constructor is private; the public
        factory resolves entries from the VERIFIED registry by id."""
        import sim.energy
        import sim.energy.bundles as bundles_mod

        assert not hasattr(bundles_mod, "machine_tuple")
        assert "machine_tuple" not in sim.energy.__all__
        assert "machine_tuple_from_registry" in sim.energy.__all__

    def test_registry_tuple_factory_rejects_unknown_entries(self, registry, mapping):
        from sim.energy.bundles import PIECEWISE

        with pytest.raises(ValueError, match="VERIFIED"):
            machine_tuple_from_registry(
                registry, mapping, node_class_name="fog-server",
                machine_id="no-such-machine", curve_family=PIECEWISE,
            )
        with pytest.raises(ValueError, match="VERIFIED"):
            machine_tuple_from_registry(
                registry, mapping, node_class_name="fog-server",
                machine_id="specpower-fujitsu-rx100s7-2011", curve_family=PIECEWISE,
                transition_entry_id="no-such-transition",
            )


class TestManifestCompleteness:
    """R2-6: every Phase-2 parameter a simulator run consumes is materialized in
    the manifest; NaN never seals; every bundle field is validated."""

    def test_manifest_has_schema_and_phase2_parameters(self, registry, mapping):
        b = build_development_bundles(registry, mapping)[0]
        man = b.manifest()
        assert man["schema"] == "energy-model-bundle-manifest-v1"
        assert man["include_image_pull_bytes"] == b.include_image_pull_bytes
        payload = man["machines"][0]["payload"]
        assert "image_pull_mib_per_s" in payload
        assert "capacity_anchor" in payload
        # ...and the emitted Phase 2 objects consume exactly these values
        coeffs = b.energy_coefficients()
        assert coeffs.include_image_pull_bytes == man["include_image_pull_bytes"]
        for m in b.machines:
            spec = m.node_class_spec()
            assert spec.image_pull_mib_per_s == m.image_pull_mib_per_s

    def test_image_pull_rate_changes_the_manifest_digest(self, registry, mapping):
        """Adversarial regression: image_pull_mib_per_s affects episode timing and
        allocated network energy but was previously invisible to the pinned hash."""
        from sim.energy.bundles import PIECEWISE, bundle_from_registry

        kwargs = dict(
            bundle_id="probe-2", role="development", curve_family=PIECEWISE,
            machines=(("fog-server", "specpower-fujitsu-rx100s7-2011", None),),
            cloud_entry_id="ccf-aws-cpu", network_entry_id="ccf-network",
        )
        a = bundle_from_registry(registry, mapping, **kwargs)
        b = bundle_from_registry(registry, mapping, image_pull_mib_per_s=50.0, **kwargs)
        c = bundle_from_registry(
            registry, mapping, include_image_pull_bytes=False, **kwargs
        )
        assert len({x.manifest_sha256() for x in (a, b, c)}) == 3

    def test_nan_never_seals(self, registry, mapping):
        from sim.energy.bundles import _canonical

        [b1, _, _] = build_development_bundles(registry, mapping)
        with pytest.raises(ValueError):
            dataclasses.replace(b1, cloud_floor_vcpu=float("nan"))
        with pytest.raises(ValueError):
            dataclasses.replace(b1, cloud_floor_vcpu=float("inf"))
        with pytest.raises(ValueError):  # the canonical serializer itself refuses NaN
            _canonical({"x": float("nan")})

    def test_cloud_floor_validated(self, registry, mapping):
        [b1, _, _] = build_development_bundles(registry, mapping)
        with pytest.raises(ValueError, match="cloud_floor_vcpu"):
            dataclasses.replace(b1, cloud_floor_vcpu=-1.0)


class TestAnchorValidation:
    """R2-8: anchors are registry-backed — domain and peak come from the cited
    entry's published values, and hand-typed divergent anchors are rejected."""

    def test_anchor_factory_uses_published_values(self, registry):
        from sim.energy.capacity import DomainAnchor

        pi = registry.arm_sbc["armsbc-rpi3b-cloutier2016"]
        a = DomainAnchor.from_machine(pi, 4000)
        assert a.domain == "hpl-gflops-cpu-dp"
        assert a.reference_peak_throughput_ops == 3.7

    def test_forged_anchor_rejected(self, registry):
        """Adversarial regression (the reviewer's demonstrated bypass, twice over):
        an RPi anchor with peak 999 previously yielded ~15 mc for the Jetson
        silently through `capacity_millicores()`. Now (a) a hand-typed anchor
        cannot even be constructed (no seal), (b) a field-edited sealed anchor is
        rejected at *mapping construction* — before any direct capacity call — and
        (c) the registry validator rejects a divergent anchor outright."""
        from sim.energy.capacity import CapacityMapping, DomainAnchor, validate_anchor

        pi = registry.arm_sbc["armsbc-rpi3b-cloutier2016"]
        with pytest.raises(ValueError, match="seal"):
            DomainAnchor(
                domain="hpl-gflops-cpu-dp",
                reference_machine_id=pi.machine_id,
                reference_peak_throughput_ops=999.0,
                reference_capacity_millicores=4000,
            )
        honest = DomainAnchor.from_machine(pi, 4000)
        forged = dataclasses.replace(honest, reference_peak_throughput_ops=999.0)
        # (b) the direct consumption path: the mapping itself refuses the anchor,
        # so mapping.capacity_millicores(jetson) is unreachable with a forgery.
        with pytest.raises(ValueError, match="seal mismatch"):
            CapacityMapping(anchors=(forged,))
        # (c) the registry check refuses it too
        with pytest.raises(ValueError, match="seal mismatch"):
            validate_anchor(forged, pi)

    def test_registry_divergent_but_sealed_anchor_rejected_by_validator(self, registry):
        """A from_machine anchor built over one entry must not validate against a
        different entry (the registry-grounding check, beyond the seal)."""
        from sim.energy.capacity import DomainAnchor, validate_anchor

        pi = registry.arm_sbc["armsbc-rpi3b-cloutier2016"]
        jetson = registry.arm_sbc["armsbc-jetsontx1-cloutier2016"]
        anchor_from_jetson = DomainAnchor.from_machine(jetson, 4000)
        with pytest.raises(ValueError, match="cites"):
            validate_anchor(anchor_from_jetson, pi)

    def test_anchor_mutation_after_mapping_construction_never_prices_capacity(self, registry):
        """R4-4 exact counterexample: mutation after mapping construction previously
        changed Jetson capacity from 17,297 to 64 mc without another seal check."""
        from sim.energy.capacity import CapacityMapping, DomainAnchor

        pi = registry.arm_sbc["armsbc-rpi3b-cloutier2016"]
        jetson = registry.arm_sbc["armsbc-jetsontx1-cloutier2016"]
        anchor = DomainAnchor.from_machine(pi, 4000)
        mapping = CapacityMapping(anchors=(anchor,))
        assert mapping.capacity_millicores(jetson) == round(4000 * 16.0 / 3.7)
        object.__setattr__(anchor, "reference_peak_throughput_ops", 999.0)
        with pytest.raises(ValueError, match="seal mismatch"):
            mapping.capacity_millicores(jetson)
        with pytest.raises(ValueError, match="seal mismatch"):
            mapping.payload()

    def test_manifest_materializes_the_anchor_payload(self, registry, mapping):
        b = build_development_bundles(registry, mapping)[0]
        far_edge = next(m for m in b.machines if m.node_class_name == "far-edge")
        payload = far_edge.payload()["capacity_anchor"]
        assert payload["reference_machine_id"] == "armsbc-rpi3b-cloutier2016"
        assert payload["reference_peak_throughput_ops"] == 3.7
        assert payload["reference_capacity_millicores"] == 4000


class TestPhase2Bridge:
    def test_node_class_specs_satisfy_w2_constraints(self, registry, mapping):
        dev = build_development_bundles(registry, mapping)
        ho = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)
        for b in dev + ho:
            for name, spec in b.node_class_specs().items():
                assert spec.suspended_power_w <= spec.power_idle_w
                assert spec.power_idle_w <= spec.max_curve_power_w()
                if name == "fog-server":
                    assert spec.suspendable  # published transition table applied
                else:
                    assert not spec.suspendable  # no published SBC transition data

    def test_energy_coefficients_carry_bundle_provenance(self, registry, mapping):
        b = build_development_bundles(registry, mapping)[0]
        c = b.energy_coefficients()
        assert b.bundle_id in c.version
        assert "ccf-aws-cpu" in c.source
        assert "swept (not CI)" in c.parameter_ranges
        # R1-4: the coefficient object names the bundle set and the manifest hash,
        # so every downstream ProvenanceRecord cites the bundle it survives.
        assert c.bundle_set == f"development:{b.bundle_id}"
        assert b.manifest_sha256() in c.source

    def test_manifest_covers_every_simulation_relevant_number(self, registry, mapping):
        """R1-4/R1-8: the pinned hash input must cover curves, capacities,
        transitions, exponents, and swept ranges — not just entry ids."""
        b = build_held_out_bundles(registry, mapping, acknowledged_one_shot=True)[0]
        man = b.manifest()
        machine = man["machines"][0]["payload"]
        for key in (
            "power_curve_points",
            "capacity_millicores",
            "suspend_energy_j",
            "wake_energy_j",
            "fit_exponent_r",
        ):
            assert key in machine
        for key in (
            "cloud_swept_j_per_vcpu_hour",
            "network_swept_j_per_byte",
            "role",
            "cloud_floor_vcpu",
        ):
            assert key in man
        # any numeric change flips the digest
        tampered = dataclasses.replace(b, cloud_floor_vcpu=b.cloud_floor_vcpu + 1.0)
        assert tampered.manifest_sha256() != b.manifest_sha256()

    def test_idle_floor_and_ceiling_from_bundle(self, registry, mapping):
        """End-to-end V1/W2 bridge: a cluster built from a bundle yields I > 0 and
        floor < ceiling under the Phase 2 primitives."""
        from sim.energy import BoundaryBounds, BoundaryWeights, ceiling_j, idle_floor_j
        from sim.spec import ClusterSpec, NodeSpec

        b = build_development_bundles(registry, mapping)[0]
        specs = b.node_class_specs()
        caps = {m.node_class_name: m.capacity_millicores for m in b.machines}
        cluster = ClusterSpec(
            classes=tuple(specs.values()),
            nodes=(
                NodeSpec(
                    id=0, node_class="fog-server", tier=1,
                    cpu_capacity=caps["fog-server"], mem_capacity=8192,
                ),
                NodeSpec(
                    id=1, node_class="far-edge", tier=2,
                    cpu_capacity=caps["far-edge"], mem_capacity=1024,
                ),
            ),
            latency_ms=((0.0, 5.0), (5.0, 0.0)),
        )
        coeffs = b.energy_coefficients()
        weights = BoundaryWeights()
        floor = idle_floor_j(cluster, coeffs, 3600.0, weights)
        ceil = ceiling_j(
            cluster, coeffs, 3600.0, weights,
            BoundaryBounds(max_transitions_per_node=4, network_bytes_bound=1e9),
        )
        assert floor > 0.0
        assert ceil > floor
