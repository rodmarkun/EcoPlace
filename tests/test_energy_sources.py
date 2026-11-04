"""Pinned-source layer tests (TODO 3.1): provenance completeness, verified-only
filtering, loader validation, and the joint-tuple discipline at the source level."""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from sim.energy.sources import (
    PROVISIONAL,
    VERIFIED,
    Provenance,
    SourceRegistry,
    SpecPowerLoadPoint,
    SpecPowerMachine,
    load_pinned_sources,
)

PROV = Provenance(
    source_name="test source",
    url="https://example.org/x",
    doi=None,
    version_or_date_accessed="2026-07-16",
    field_transcribed="test field",
    status=VERIFIED,
    verification_method="constructed in test",
)


def _machine(status: str = VERIFIED, **overrides) -> SpecPowerMachine:
    points = tuple(
        SpecPowerLoadPoint(
            target_load_frac=(i + 1) / 10.0,
            actual_load_frac=(i + 1) / 10.0,
            ssj_ops=1000.0 * (i + 1),
            avg_active_power_w=100.0 + 20.0 * (i + 1),
        )
        for i in range(10)
    )
    kwargs = dict(
        machine_id="specpower-test-machine",
        vendor="V",
        model="M",
        cpu="C",
        cpu_enabled="8 cores",
        memory_gb="16",
        test_date="Jan 1, 2020",
        hardware_availability="Jan-2020",
        publication="Feb 1, 2020",
        overall_ssj_ops_per_watt=1000.0,
        load_points=points,
        active_idle_w=50.0,
        provenance=dataclasses.replace(PROV, status=status),
    )
    kwargs.update(overrides)
    return SpecPowerMachine(**kwargs)


class TestProvenance:
    def test_requires_url_or_doi(self):
        with pytest.raises(ValueError, match="URL or a DOI"):
            dataclasses.replace(PROV, url=None, doi=None)

    def test_rejects_unknown_status(self):
        with pytest.raises(ValueError, match="status"):
            dataclasses.replace(PROV, status="probably fine")

    def test_requires_field_transcribed(self):
        with pytest.raises(ValueError, match="field_transcribed"):
            dataclasses.replace(PROV, field_transcribed="")

    def test_provisional_is_not_verified(self):
        assert not dataclasses.replace(PROV, status=PROVISIONAL).is_verified
        assert PROV.is_verified


class TestSpecPowerMachine:
    def test_joint_curve_starts_at_idle_and_spans_unit_load(self):
        m = _machine()
        pts = m.joint_curve_points()
        assert pts[0] == (0.0, 50.0)
        assert pts[-1][0] == 1.0
        assert len(pts) == 11

    def test_rejects_non_ascending_throughput(self):
        pts = list(_machine().load_points)
        pts[5] = dataclasses.replace(pts[5], ssj_ops=pts[4].ssj_ops)  # tie -> not ascending
        with pytest.raises(ValueError, match="ascend"):
            _machine(load_points=tuple(pts))

    def test_rejects_idle_at_or_above_low_load_power(self):
        with pytest.raises(ValueError, match="idle"):
            _machine(active_idle_w=120.0)

    def test_rejects_wrong_level_count(self):
        with pytest.raises(ValueError, match="10 load levels"):
            _machine(load_points=_machine().load_points[:9])


class TestRegistry:
    def test_verified_filter_drops_provisional_entries(self, tmp_path):
        from sim.energy.sources import DATA_DIR

        data = tmp_path / "data"
        shutil.copytree(DATA_DIR, data)
        path = data / "specpower.json"
        payload = json.loads(path.read_text())
        provisional_id = next(iter(payload["entries"]))
        payload["entries"][provisional_id]["provenance"]["status"] = PROVISIONAL
        path.write_text(json.dumps(payload))
        reg = load_pinned_sources(data)
        assert provisional_id in reg.specpower
        assert provisional_id not in reg.verified().specpower

    def test_duplicate_ids_rejected_across_groups(self, tmp_path):
        from sim.energy.sources import DATA_DIR

        data = tmp_path / "data"
        shutil.copytree(DATA_DIR, data)
        path = data / "pinned_sources.json"
        payload = json.loads(path.read_text())
        duplicate_id = next(iter(json.loads((data / "specpower.json").read_text())["entries"]))
        duplicate = dict(payload["arm_sbc"][0])
        duplicate["machine_id"] = duplicate_id
        payload["arm_sbc"].append(duplicate)
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="duplicate"):
            load_pinned_sources(data)

    def test_caller_built_verified_registry_cannot_impersonate_pinned_data(self):
        """R4-3 exact counterexample: a caller-modified Lenovo object carrying the
        original VERIFIED label must not become source authority by wrapping it."""
        pinned = load_pinned_sources()
        machine_id = "specpower-lenovo-sr665v3-2023"
        modified = dataclasses.replace(
            pinned.specpower[machine_id],
            active_idle_w=pinned.specpower[machine_id].active_idle_w + 1.0,
        )
        with pytest.raises(ValueError, match="loader-issued"):
            SourceRegistry(specpower={machine_id: modified})

        # Even deliberate use of the internal issuing helper cannot turn a modified
        # object into canonical data by copying the canonical file digest.
        forged = SourceRegistry._issue(
            specpower={**pinned.specpower, machine_id: modified},
            arm_sbc=pinned.arm_sbc,
            transitions=pinned.transitions,
            cloud_allocation=pinned.cloud_allocation,
            network_allocation=pinned.network_allocation,
            source_data_sha256=pinned.source_data_sha256,
        )
        forged.verify_seal()  # internally consistent, but not repository-canonical
        with pytest.raises(ValueError, match="payload differs"):
            forged.require_canonical()

    def test_loaded_registry_detects_post_load_entry_mutation(self):
        reg = load_pinned_sources()
        machine = reg.specpower["specpower-lenovo-sr665v3-2023"]
        object.__setattr__(machine, "active_idle_w", machine.active_idle_w + 1.0)
        with pytest.raises(ValueError, match="seal mismatch"):
            reg.verified()


class TestPinnedData:
    """The actual shipped data files must load, validate, and be fully attributed."""

    def test_loads_and_every_entry_has_full_provenance(self):
        reg = load_pinned_sources()
        entries = reg.all_entries()
        assert len(reg.specpower) >= 6
        assert len(reg.arm_sbc) >= 2
        assert len(reg.transitions) >= 2
        assert len(reg.cloud_allocation) >= 2
        assert len(reg.network_allocation) >= 2
        for entry in entries.values():
            p = entry.provenance
            assert p.url or p.doi
            assert p.version_or_date_accessed
            assert p.field_transcribed
            assert p.verification_method

    def test_specpower_entries_are_single_node_joint_tables(self):
        reg = load_pinned_sources()
        for m in reg.specpower.values():
            assert len(m.load_points) == 10
            ops = [p.ssj_ops for p in m.load_points]
            assert ops == sorted(ops)
            assert m.active_idle_w < m.load_points[0].avg_active_power_w
            # provenance must point at the actual spec.org result page
            assert "spec.org/power_ssj2008/results" in m.provenance.url

    def test_specpower_json_declares_machine_parsed_transcription(self):
        reg = load_pinned_sources()
        for m in reg.specpower.values():
            assert m.provenance.status == VERIFIED
            assert "fetch_specpower.py" in m.provenance.verification_method

    def test_no_invented_ranges(self):
        """Sources publishing only a point estimate must say so; swept ranges must
        bracket the point (V2 of the data-integrity rule: sweeps, never CIs)."""
        reg = load_pinned_sources()
        for c in reg.cloud_allocation.values():
            lo, hi = c.swept_range_w_per_vcpu
            assert lo <= c.point_w_per_vcpu <= hi
        for n in reg.network_allocation.values():
            lo, hi = n.swept_range_kwh_per_gb
            assert lo <= n.kwh_per_gb <= hi

    def test_raw_json_provenance_statuses_are_canonical(self):
        from sim.energy.sources import DATA_DIR

        def statuses(node):
            if isinstance(node, dict):
                if "status" in node and "field_transcribed" in node:
                    yield node["status"]
                for v in node.values():
                    yield from statuses(v)
            elif isinstance(node, list):
                for v in node:
                    yield from statuses(v)

        for name in ("specpower.json", "pinned_sources.json"):
            payload = json.loads((DATA_DIR / name).read_text())
            found = list(statuses(payload))
            assert found, f"{name}: no provenance records found"
            assert all(s in (VERIFIED, PROVISIONAL) for s in found)

    def test_unit_conversions(self):
        reg = load_pinned_sources()
        ccf = reg.network_allocation["ccf-network"]
        # 0.001 kWh/GB = 0.001 * 3.6e6 J / 1e9 B = 3.6e-6 J/B
        assert ccf.j_per_byte == pytest.approx(3.6e-6)
        aws = reg.cloud_allocation["ccf-aws-cpu"]
        # 0.74 + 0.5*(3.5-0.74) = 2.12 W/vCPU -> 7632 J per vCPU-hour
        assert aws.point_w_per_vcpu == pytest.approx(2.12)
        assert aws.point_j_per_vcpu_hour == pytest.approx(7632.0)


class TestRetainedArtifacts:
    """R2-2: recorded content hashes are *verified* against the retained raw
    artifacts at every load — never merely asserted."""

    def test_ccf_entries_cite_the_retained_artifact(self):
        reg = load_pinned_sources()
        for eid in ("ccf-aws-cpu", "ccf-gcp-cpu"):
            p = reg.cloud_allocation[eid].provenance
            assert p.artifact_path == "artifacts/ccf-methodology-2026-07-18.html"
            assert p.content_sha256
        assert (
            reg.network_allocation["ccf-network"].provenance.artifact_path
            == "artifacts/ccf-methodology-2026-07-18.html"
        )

    def test_corrupted_artifact_refused_at_load(self, tmp_path):
        """Adversarial regression: previously a modified retained artifact loaded
        silently — the recorded hash was decorative."""
        import shutil

        from sim.energy.sources import DATA_DIR

        d = tmp_path / "data"
        shutil.copytree(DATA_DIR, d)
        art = d / "artifacts" / "ccf-methodology-2026-07-18.html"
        art.write_bytes(art.read_bytes() + b"<!-- tampered -->")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_pinned_sources(d)

    def test_missing_artifact_refused_at_load(self, tmp_path):
        import shutil

        from sim.energy.sources import DATA_DIR

        d = tmp_path / "data"
        shutil.copytree(DATA_DIR, d)
        (d / "artifacts" / "ccf-methodology-2026-07-18.html").unlink()
        with pytest.raises(ValueError, match="does not exist"):
            load_pinned_sources(d)

    def test_artifact_path_requires_content_hash(self):
        with pytest.raises(ValueError, match="content_sha256"):
            dataclasses.replace(PROV, artifact_path="artifacts/x.html")

    def test_artifact_path_containment(self, tmp_path):
        """R3-6 adversarial regression: an absolute or parent-traversing
        artifact_path could name (and 'verify') a file outside the pinned data
        directory. Both are rejected structurally, and resolution is re-checked."""
        with pytest.raises(ValueError, match="relative"):
            dataclasses.replace(
                PROV, artifact_path="/etc/hosts", content_sha256="0" * 64
            )
        with pytest.raises(ValueError, match="relative"):
            dataclasses.replace(
                PROV, artifact_path="../outside.html", content_sha256="0" * 64
            )
        # resolution containment (symlink escape) is re-checked at verify time
        outside = tmp_path / "outside.html"
        outside.write_bytes(b"outside")
        data_dir = tmp_path / "data"
        (data_dir / "artifacts").mkdir(parents=True)
        (data_dir / "artifacts" / "link.html").symlink_to(outside)
        import hashlib as _h

        prov = dataclasses.replace(
            PROV,
            artifact_path="artifacts/link.html",
            content_sha256=_h.sha256(b"outside").hexdigest(),
        )
        with pytest.raises(ValueError, match="escapes"):
            prov.verify_retained_artifact(data_dir)


class TestBoaviztaScopeDeviation:
    """R2-7: Boavizta publishes a nonlinear instance-specific consumption-profile
    model, not a scalar linear-interpolation coefficient — so no cloud_allocation
    entry models it, consistently with the README and the review log. The raw API
    response is retained as evidence for the explicit P2 bundle-family decision."""

    def test_no_boavizta_scalar_entry_is_pinned(self):
        reg = load_pinned_sources()
        assert not any("boavizta" in k.lower() for k in reg.all_entries())

    def test_deviation_documented_with_retained_evidence(self):
        import hashlib

        from sim.energy.sources import DATA_DIR

        payload = json.loads((DATA_DIR / "pinned_sources.json").read_text())
        note = payload["boavizta_scope_deviation"]
        assert "consumption-profile" in note
        assert "P2" in note
        art = DATA_DIR / "artifacts" / "boaviztapi-aws-m5xlarge-2026-07-18.json"
        assert art.is_file()
        assert hashlib.sha256(art.read_bytes()).hexdigest() in note
