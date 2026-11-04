"""Synthetic evidence tests; no simulator episode, policy execution or model call."""

import copy
import dataclasses
import json

import pytest

from evolution.evaluation import config_for_development_bundle
from experiments.gp_tuning_20260902.recovery import digest
from experiments.validation_20260909 import run_independent_calibration as runner
from experiments.validation_20260909 import summarize_independent_calibration as summary
from sim.energy import build_development_bundles, load_pinned_sources
from sim.energy.boundary import BoundaryBounds, BoundaryWeights, EnergyMeter, certify_boundary
from sim.energy.fitness import ConstraintSpec, InstanceOutcome, evaluate_candidate
from sim.spec import ArrivalSpec, config_sha256, load_episode_config


@pytest.fixture(scope="module")
def design():
    return json.loads((runner.design_module.OUT / "DESIGN.json").read_text())


@pytest.fixture(scope="module")
def config(design):
    base = load_episode_config(runner.design_module.OUT / (design["cells"][0]["id"] + ".json"))
    return config_for_development_bundle(base, build_development_bundles(load_pinned_sources())[0])


def outcomes(design):
    return [
        {"job": job, "status": "evaluated", "feasible": False, "normalized_energy": None}
        for job in runner.jobs(design, "calibration")
    ]


def set_feasible(rows, design, *, pressure, label=None, profile=None, topology=None):
    cells = {
        c["id"]
        for c in design["cells"]
        if c["nominal_average_edge_cpu_pressure"] == pressure
        and (profile is None or c["profile"] == profile)
        and (topology is None or c["topology"] == topology)
    }
    for row in rows:
        if row["job"]["cell_id"] in cells and (label is None or row["job"]["label"] == label):
            row["feasible"] = True
            row["normalized_energy"] = 1.0


def test_preference_and_no_energy_based_selection(design):
    rows = outcomes(design)
    label = design["fixed_references"][0]["label"]
    set_feasible(rows, design, pressure=0.15)
    set_feasible(rows, design, pressure=0.30, label=label)
    report = summary.admission(design, rows)
    assert all(r["selected_pressure"] == 0.30 for r in report["topologies"].values())
    assert all(
        r["qualifying_references_by_pressure"]["0.30"] == [label]
        for r in report["topologies"].values()
    )
    for row in rows:
        row["normalized_energy"] = 9999 if row["feasible"] else None
    assert summary.admission(design, rows) == report


def test_references_cannot_be_stitched_into_an_oracle(design):
    rows = outcomes(design)
    a, b = [p["label"] for p in design["fixed_references"][:2]]
    set_feasible(rows, design, pressure=0.30, label=a, profile="steady")
    set_feasible(rows, design, pressure=0.30, label=b, profile="two_bursts")
    set_feasible(rows, design, pressure=0.15, label=a)
    result = summary.admission(design, rows)
    assert all(t["selected_pressure"] == 0.15 for t in result["topologies"].values())
    assert all(
        not t["qualifying_references_by_pressure"]["0.30"] for t in result["topologies"].values()
    )


def test_stress_is_not_a_fallback_and_fault_is_not_feasibility(design):
    rows = outcomes(design)
    set_feasible(rows, design, pressure=0.45)
    assert summary.admission(design, rows)["status"] == "stop-design-amendment-required"
    label = design["fixed_references"][0]["label"]
    set_feasible(rows, design, pressure=0.30, label=label)
    row = next(
        r
        for r in rows
        if r["feasible"] and r["job"]["label"] == label and "0.30" in r["job"]["cell_id"]
    )
    row.update(status="policy-fault", feasible=None, normalized_energy=None)
    assert summary.admission(design, rows)["status"] == "stop-design-amendment-required"


def test_missing_duplicate_and_unknown_outcomes_refused(design):
    rows = outcomes(design)
    for bad in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="complete unique"):
            summary.admission(design, bad)
    rows[0]["status"] = "worker-timeout"
    with pytest.raises(ValueError, match="technical failure"):
        summary.admission(design, rows)


ROWS = [
    [0, 0, 30.0, 30.1, "completed"],
    [1, 0, 31.0, 31.5, "completed"],
    [2, 1, 32.0, None, "in_flight"],
    [3, 1, 33.0, 33.1, "dropped_queue_full"],
    [4, 2, 34.0, 34.4, "timeout"],
    [5, 2, 35.0, 35.2, "completed"],
]


def test_offered_denominators_and_deterministic_compression(tmp_path, config):
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    metadata = runner.write_offered_trace(a, ROWS)
    runner.write_offered_trace(b, iter(ROWS))
    assert a.read_bytes() == b.read_bytes()
    out = summary.cohort(a, metadata, config)
    assert out["offered"] == 6 and out["goodput"] == 0.5
    assert out["drop_rate"] == out["timeout_rate"] == 1 / 6
    assert [t["slo_achieved"] for t in out["per_template"].values()] == [0.5, 0, 0.5]
    assert out["statuses"]["in_flight"] == 1


@pytest.mark.parametrize(
    "edit",
    ["id", "template", "order", "cutoff", "terminal", "unfinished", "status", "count", "digest"],
)
def test_corrupt_offered_cohort_refused(tmp_path, config, edit):
    rows = copy.deepcopy(ROWS)
    if edit == "id":
        rows[1][0] = 0
    if edit == "template":
        rows[0][1] = True
    if edit == "order":
        rows[1][2] = 29.0
    if edit == "cutoff":
        rows[-1][2] = 870.0
        rows[-1][3] = 870.1
    if edit == "terminal":
        rows[0][3] = 29.9
    if edit == "unfinished":
        rows[2][3] = 40.0
    if edit == "status":
        rows[0][4] = "silently_omitted"
    path = tmp_path / "trace.gz"
    meta = runner.write_offered_trace(path, rows)
    if edit == "count":
        meta["offered_records"] -= 1
    if edit == "digest":
        meta["uncompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        summary.cohort(path, meta, config)


def test_seed_stream_hash_is_independent_of_terminal_outcome(tmp_path, config):
    templates = []
    for t in config.workload.templates:
        times = tuple(r[2] for r in ROWS if r[1] == t.id)
        templates.append(
            dataclasses.replace(
                t,
                arrival=ArrivalSpec(
                    "poisson", 1, start_s=30, realized_times_s=times, arrivals_realized=True
                ),
            )
        )
    fixed = dataclasses.replace(
        config, workload=dataclasses.replace(config.workload, templates=tuple(templates))
    )
    path = tmp_path / "trace.gz"
    meta = runner.write_offered_trace(path, ROWS)
    observed = summary.cohort(path, meta, fixed)
    actual = {
        name: {k: values[k] for k in ("offered", "arrival_sha256")}
        for name, values in observed["per_template"].items()
    }
    assert summary.expected_arrivals(fixed, 123) == actual
    changed = copy.deepcopy(ROWS)
    changed[0][3] += 0.1
    path2 = tmp_path / "changed.gz"
    meta2 = runner.write_offered_trace(path2, changed)
    assert (
        summary.cohort(path2, meta2, fixed)["per_template"]["interactive"]["arrival_sha256"]
        == actual["interactive"]["arrival_sha256"]
    )


def synthetic_evidence(config, design, observed, job=None, plan=None):
    # Meter arithmetic only: these are manufactured unit-test outcomes, not simulated requests.
    job = job or runner.jobs(design, "calibration")[0]
    req = runner.request(plan, job) if plan else {"plan_sha256": "test-only", "job": job}
    weights = BoundaryWeights(**design["boundary_weights"])
    bounds = BoundaryBounds(**design["boundary_bounds"])
    cert = certify_boundary(
        config.cluster,
        config.energy,
        config.horizon_s,
        weights,
        bounds,
        scenario_id=job["cell_id"],
        seed=job["seed"],
        bundle_id=job["bundle_id"],
    )
    energy = EnergyMeter(config.cluster, config.energy).finish(config.horizon_s)
    token = cert.verify_covers(energy)
    slo = {n: v["slo_achieved"] for n, v in observed["per_template"].items()}
    disruption = dict.fromkeys(design["constraints"]["disruption_caps"], 0.0)
    outcome = InstanceOutcome(
        token, slo, observed["goodput"], observed["drop_rate"], observed["timeout_rate"], disruption
    )
    fitness = evaluate_candidate(
        ConstraintSpec(**design["constraints"]),
        [outcome],
        development_bundle_ids=(job["bundle_id"],),
        instance_keys=((job["cell_id"], job["seed"]),),
        expected_certificate_seals={(job["cell_id"], job["seed"], job["bundle_id"]): cert.seal},
        proposal_seq=0,
    )
    ledger = {
        "requests_injected": observed["offered"],
        **{"requests_" + s: observed["statuses"].get(s, 0) for s in summary.STATUSES[:-1]},
    }
    instance = {
        "scenario_id": job["cell_id"],
        "seed": job["seed"],
        "bundle_id": job["bundle_id"],
        "episode_digest": "test-only-not-a-runtime-digest",
        "certificate_seal": cert.seal,
        "energy_j": token.energy_j,
        "idle_floor_j": token.idle_floor_j,
        "ceiling_j": token.ceiling_j,
        "slo_achieved": slo,
        "goodput": outcome.goodput,
        "drop_rate": outcome.drop_rate,
        "timeout_rate": outcome.timeout_rate,
        "disruption": disruption,
        "boundary_certificate": {**cert.payload(), "seal": cert.seal},
        "verified_energy": {**token.payload(), "seal": token.seal},
        "episode": {
            "config_sha256": config_sha256(config),
            "seed": job["seed"],
            "horizon_s": config.horizon_s,
            "records": {"length": observed["offered"]},
            "ledger": ledger,
            "rebalancer": disruption,
            "energy": energy.payload(),
        },
    }
    f = dataclasses.asdict(fitness)
    f["infeasibility_vector"] = list(fitness.infeasibility_vector)
    f.update(sort_key=list(fitness.sort_key()), direction="lower-is-better")
    result = {
        "request_sha256": digest(req),
        "effective_config_sha256": config_sha256(config),
        "status": "evaluated",
        "evaluation": {
            "fitness": f,
            "policy": {"artifact_sha256": job["source_sha256"]},
            "instances": [instance],
        },
    }
    return req, result


@pytest.mark.parametrize(
    "tamper", [None, "fitness", "slo", "certificate", "energy", "negative", "ledger"]
)
def test_recomputed_fitness_energy_and_counts(tmp_path, config, design, tamper):
    path = tmp_path / "trace.gz"
    meta = runner.write_offered_trace(path, ROWS)
    observed = summary.cohort(path, meta, config)
    req, result = synthetic_evidence(config, design, observed)
    i = result["evaluation"]["instances"][0]
    if tamper == "fitness":
        result["evaluation"]["fitness"]["feasible"] = True
    if tamper == "slo":
        i["slo_achieved"]["interactive"] = 1.0
    if tamper == "certificate":
        i["boundary_certificate"]["weights"][0] = 2
    if tamper == "energy":
        i["episode"]["energy"]["device_j"] += 1
    if tamper == "negative":
        i["episode"]["energy"]["cloud_j"] = -1
    if tamper == "ledger":
        i["episode"]["ledger"]["requests_injected"] -= 1
    if tamper:
        with pytest.raises(ValueError):
            summary.verify_measured(req, result, config, design, observed)
    else:
        row = summary.verify_measured(req, result, config, design, observed)
        assert row["feasible"] is False and row["normalized_energy"] is None


def test_missing_completion_is_refused_before_outcome_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner, "verify_source", lambda out: pytest.fail("partial source/outcome read")
    )
    with pytest.raises(FileNotFoundError):
        summary.verify(tmp_path)


def test_pilot_fault_cannot_supply_a_resource_projection(design):
    plan = {"design": design}
    phase = {"rows": [{"status": "evaluated"}] * 3, "files_sha256": {}}
    assert summary.pilot_closure(plan, phase)["requests"] == 3
    phase["rows"] = [{"status": "evaluated"}] * 2 + [{"status": "policy-fault"}]
    with pytest.raises(ValueError, match="three measured episodes"):
        summary.pilot_closure(plan, phase)


def test_fault_cannot_invent_feasibility(design):
    rows = outcomes(design)
    rows[0].update(status="policy-fault", feasible=False)
    with pytest.raises(ValueError, match="must not invent"):
        summary.admission(design, rows)


def test_complete_on_disk_grid_and_pilot_dispatch_boundary(tmp_path, monkeypatch, design):
    # All 975 jobs and files exist, but request outcomes are manufactured. Only
    # source/environment verification and seeded arrival generation are mocked.
    # No policy is executed, no simulator is called, and no scientific result exists.
    from contextlib import nullcontext

    from experiments.gp_tuning_20260902.recovery import read_sealed

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"payload": value, "payload_sha256": digest(value)}))

    plan = {
        "design": design,
        "pilot_jobs": runner.jobs(design, "pilot"),
        "calibration_jobs": runner.jobs(design, "calibration"),
        "rss_limit_mib": 2048,
        "worker_timeout_seconds": 1200,
        "pilot_total_seconds": 3600,
        "calibration_total_seconds": 21600,
        "projection_reserve_factor": 1.30,
        "source_archive_sha256": "explicit-test-only-source-placeholder",
    }
    write(tmp_path / "snapshot.json", {})
    identities = {}
    for policy in design["fixed_references"]:
        path = tmp_path / "policies" / (policy["source_sha256"] + ".ps")
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((runner.ROOT / policy["path"]).read_bytes())
        identities[policy["label"]] = summary.policy_identity_for_source(path.read_text())
    bundles = {b.bundle_id: b for b in build_development_bundles(load_pinned_sources())}
    configs, usages, expected_arrivals = {}, {}, {}
    rows = [
        [0, 0, 30.0, 30.01, "completed"],
        [1, 1, 31.0, 31.01, "completed"],
        [2, 2, 32.0, 32.01, "completed"],
    ]
    for phase in ("pilot", "calibration"):
        usages[phase] = []
        for index, job in enumerate(plan[phase + "_jobs"]):
            ck = (job["cell_id"], job["bundle_id"])
            if ck not in configs:
                base = load_episode_config(runner.design_module.OUT / (job["cell_id"] + ".json"))
                configs[ck] = config_for_development_bundle(base, bundles[job["bundle_id"]])
            config = configs[ck]
            req = runner.request(plan, job)
            key = digest(req)
            folder = tmp_path / phase
            folder.mkdir(exist_ok=True)
            if phase == "calibration" and index == 1:
                result = {
                    "request_sha256": key,
                    "effective_config_sha256": job["effective_config_sha256"],
                    "status": "policy-fault",
                    "failure": {
                        "scenario_id": job["cell_id"],
                        "seed": job["seed"],
                        "bundle_id": job["bundle_id"],
                        "message": "test-only fault",
                    },
                }
            else:
                records = copy.deepcopy(rows)
                if phase == "calibration" and index == 0:
                    records[0][3:] = [None, "in_flight"]
                trace = folder / (key + ".records.jsonl.gz")
                meta = runner.write_offered_trace(trace, records)
                observed = summary.cohort(trace, meta, config)
                expected_arrivals[(job["cell_id"], job["seed"])] = {
                    n: {k: v[k] for k in ("offered", "arrival_sha256")}
                    for n, v in observed["per_template"].items()
                }
                _, result = synthetic_evidence(config, design, observed, job, plan)
                result["evaluation"]["policy"] = identities[job["label"]]
                result["offered_trace"] = meta
            write(folder / (key + ".request.json"), req)
            write(
                folder / (key + ".started.json"),
                {"request_sha256": key, "pid": 1, "started_at_utc": "test-only"},
            )
            write(folder / (key + ".result.json"), result)
            (folder / (key + ".log")).write_text("manufactured integration evidence\n")
            usage = {
                "request_sha256": key,
                "result_sha256": digest(result),
                "started_file_sha256": runner.sha(folder / (key + ".started.json")),
                "elapsed_seconds": 1.0,
                "peak_rss_mib": 10.0,
                "timeout_seconds": 1200.0,
                "exit_code": 0,
                "stop_reason": None,
            }
            write(folder / (key + ".usage.json"), usage)
            usages[phase].append(usage)
    monkeypatch.setattr(runner, "verify_source", lambda out: plan)
    monkeypatch.setattr(runner, "verify_archive", lambda *args: None)
    monkeypatch.setattr(runner, "shared_lock", lambda path: nullcontext())
    config_cells = {config_sha256(c): ck[0] for ck, c in configs.items()}
    monkeypatch.setattr(
        summary,
        "expected_arrivals",
        lambda c, seed: expected_arrivals[(config_cells[config_sha256(c)], seed)],
    )
    pilot = summary.verify_phase(tmp_path, plan, "pilot")
    write(tmp_path / "PILOT_CLOSURE.json", summary.pilot_closure(plan, pilot))
    write(tmp_path / "PILOT_REVIEW.json", runner.pilot_review(plan, usages["pilot"]))
    complete = {
        "plan_sha256": digest(plan),
        "requests": 972,
        "usage_sha256": [digest(u) for u in usages["calibration"]],
        "total_worker_seconds": 972.0,
        "peak_rss_mib": 10.0,
        "no_model_calls": True,
    }
    write(tmp_path / "COMPLETE.json", complete)
    report = summary.verify(tmp_path)
    assert len(report["rows"]) == 972 and len(report["groups"]) == 108
    assert report["rows"][0]["feasible"] is False
    assert report["rows"][1]["status"] == "policy-fault" and report["rows"][1]["feasible"] is None
    assert sum(r["feasible"] is True for r in report["rows"]) == 970
    assert all(t["selected_pressure"] == 0.30 for t in report["admission"]["topologies"].values())
    assert "All 972 calibration episodes" in summary.render(report)
    extra = tmp_path / "pilot" / "unexpected.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="extra or missing"):
        summary.verify_phase(tmp_path, plan, "pilot")
    extra.unlink()
    write(tmp_path / "COMPLETE.json", {**complete, "requests": 971})
    with pytest.raises(ValueError, match="completion identity or count"):
        summary.verify(tmp_path)
    (tmp_path / "COMPLETE.json").unlink()
    # Reseal coherent file bindings around false QoS; semantic verification must
    # stop run() before the first calibration dispatch, despite successful usage.
    req = runner.request(plan, plan["pilot_jobs"][0])
    key = digest(req)
    result_path = tmp_path / "pilot" / (key + ".result.json")
    result = read_sealed(result_path)
    result["evaluation"]["instances"][0]["slo_achieved"]["interactive"] = 0.0
    write(result_path, result)
    usages["pilot"][0]["result_sha256"] = digest(result)
    write(tmp_path / "pilot" / (key + ".usage.json"), usages["pilot"][0])
    calls = []

    def cached(out, req, plan, remaining):
        assert req["job"]["phase"] == "pilot", "calibration dispatched before valid pilot closure"
        calls.append(req["job"])
        return read_sealed(tmp_path / "pilot" / (digest(req) + ".usage.json"))

    monkeypatch.setattr(runner, "guarded", cached)
    with pytest.raises(ValueError, match="SLO attainment"):
        runner.run(tmp_path, pilot_only=False)
    assert len(calls) == 3
