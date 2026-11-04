"""G0 throughput pilots and literal arm-level capacity accounting.

The module lives beside :mod:`evolution.ladder` because G0 times the exact Phase-5.4
rung implementations used by every Phase-7 PolicyBundle adapter.  Development-test
runs are permanently non-evidentiary and can emit only ``DEVELOPMENT_*`` statuses.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from baselines.handwritten import all_handwritten
from sim.energy import build_development_bundles, load_pinned_sources
from sim.energy.gsa import estimate_compute_cost, stratum_for_bundle

from .codex_engine import A1_REFERENCE, CODEX_ENGINE_NAME, CODEX_MODEL_ID
from .evaluation import TaskEvaluationConfig, load_task_evaluation_config
from .ladder import (
    LadderConfig,
    _run_l0_adapter,
    _run_l1_adapter,
    load_ladder_config,
)

ROOT = Path(__file__).resolve().parent.parent
TASK_CONFIG = ROOT / "evolution/task/config.json"
CAMPAIGN_CONFIG = ROOT / "evolution/campaigns.json"
CAPACITY_SCHEMA = "evolved-continuum-g0-capacity-table-v2"
TIMING_SCHEMA = "evolved-continuum-g0-rung-timing-pilot-v1"
CEILINGS_SCHEMA = "evolved-continuum-g0-ceilings-v1"
LLM_PILOT_SCHEMA = "evolved-continuum-g0-llm-generation-pilot-v2"
DEVELOPMENT_EVIDENCE_CLASS = "DEVELOPMENT_TEST_LOAD_CONTAMINATED"
GATE_EVIDENCE_CLASS = "G0_GATE_EVIDENCE_QUIET_HOST"
LLM_ENGINE_DISPLAY = "codex-cli-subscription (A1)"
LLM_PILOT_MISSING = "CODEX_CLI_PILOT_NOT_SUPPLIED"
LLM_BLOCKING_REASON = "P1_CONFIRMATORY_ENGINE_DECISION_NOT_REGISTERED_A1"
PLACEHOLDER_CEILINGS_STATUS = "PLACEHOLDER_RESEARCHER_INPUT_REQUIRED"
APPROVED_CEILINGS_STATUS = "RESEARCHER_APPROVED_FOR_G0"
RESERVE_FRACTION = Decimal("0.30")
MAX_GATE_NORMALIZED_LOAD_1M = 0.70

EXPECTED_TASK_SHA256 = "532ba8991a2baa0e33c60ed19a47ff3d8a9f39037be1b6cafbb05288f9a9d41a"
EXPECTED_SCENARIO = "dev-small-cluster"
EXPECTED_SCENARIO_SEEDS = (104729, 130363)
EXPECTED_BUNDLES = ("dev-1", "dev-2", "dev-3")

DEFAULT_PILOT_SEEDS = (
    49979687,
    86028121,
    122949829,
    160481219,
    198491317,
    236887699,
    275604541,
    314606891,
    353868013,
    393342739,
    433024253,
    472882049,
    512927377,
    553105243,
    593441843,
    633910099,
    674506081,
    715225739,
    756065159,
    797003413,
    838041647,
    879190841,
    920419813,
    961748941,
    1003181017,
    1044764063,
    1086464111,
    1128286889,
    1170229849,
    1212300157,
)

MIN_GATE_REPETITIONS = {"L0": 30, "L1": 10, "L2": 5}
CAPACITY_FAILURE_RATE_FLOOR = 0.05
STORAGE_MB_PER_INSTANCE = {"L0": 0.01, "L1": 0.25, "L2": 2.0}
RESEARCHER_HOURS_PER_ARM = 1.0
RESEARCHER_HOURS_PER_CAMPAIGN = 0.5
GSA_RESEARCHER_HOURS = 4.0
GSA_STORAGE_MB_PER_EVALUATION = 0.002
DEFAULT_PARALLEL_CPU_WORKERS = 4


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    experiment: str
    method: str
    workload: str
    campaigns: int
    expected_candidates_per_campaign: int
    scenarios: int
    seeds: int
    bundles: int
    generation_required: bool
    role: str
    cut_order: str | None = None

    def __post_init__(self) -> None:
        if self.workload not in {"ladder", "l2-only"}:
            raise ValueError(f"{self.arm_id}: unknown workload {self.workload!r}")
        for name in (
            "campaigns",
            "expected_candidates_per_campaign",
            "scenarios",
            "seeds",
            "bundles",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{self.arm_id}: {name} must be a positive int")


PROVISIONAL_ARMS = (
    ArmSpec(
        "e1-llm-evolution",
        "E1",
        "LLM evolution",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        True,
        "confirmatory reusable-policy competitor",
    ),
    ArmSpec(
        "e1-same-dsl-gp",
        "E1/E5-1",
        "same-DSL GP operator-controlled ablation",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        False,
        "confirmatory; never descoped",
    ),
    ArmSpec(
        "e1-classic-gphh",
        "E1",
        "classic strongly-typed GPHH",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        False,
        "confirmatory reusable-policy competitor",
    ),
    ArmSpec(
        "e1-ppo-frozen",
        "E1",
        "PPO frozen policy",
        "l2-only",
        10,
        1,
        3,
        4,
        3,
        False,
        "confirmatory evaluation-only cost; training priced separately at P2",
    ),
    ArmSpec(
        "e1-random-ast-floor",
        "E1",
        "random-AST mutation floor",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        False,
        "confirmatory sanity control",
    ),
    *(
        ArmSpec(
            f"e1-reference-{name.replace('_', '-')}",
            "E1",
            f"hand-written reference: {name}",
            "l2-only",
            1,
            1,
            3,
            4,
            3,
            False,
            "evaluation-only reference cost",
        )
        for name in (
            "latency_greedy",
            "energy_greedy",
            "first_fit",
            "least_allocated",
            "most_allocated",
            "network_overhead",
        )
    ),
    ArmSpec(
        "e3-continuous-llm-adaptation",
        "E3",
        "continuous LLM adaptation at provisional triggers",
        "ladder",
        30,
        10,
        3,
        4,
        3,
        True,
        "confirmatory online comparator search cost",
    ),
    ArmSpec(
        "e3-online-gphh-adaptation",
        "E3",
        "online-GPHH at identical provisional triggers/budget",
        "ladder",
        30,
        10,
        3,
        4,
        3,
        False,
        "confirmatory online comparator search cost",
    ),
    ArmSpec(
        "e3-episodes-continuous-llm",
        "E3",
        "continuous-loop outcome episodes",
        "l2-only",
        30,
        1,
        3,
        10,
        3,
        False,
        "confirmatory RQ2 episode cost",
    ),
    ArmSpec(
        "e3-episodes-frozen-champion",
        "E3",
        "frozen-champion outcome episodes",
        "l2-only",
        30,
        1,
        3,
        10,
        3,
        False,
        "confirmatory RQ2 no-adaptation control",
    ),
    ArmSpec(
        "e3-episodes-online-gphh",
        "E3",
        "online-GPHH outcome episodes",
        "l2-only",
        30,
        1,
        3,
        10,
        3,
        False,
        "confirmatory RQ2 episode cost",
    ),
    ArmSpec(
        "e5-2-text-feedback-off",
        "E5-2",
        "text feedback off",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        True,
        "enumerated ablation contrast",
    ),
    ArmSpec(
        "e5-3-numeric-patching-off",
        "E5-3",
        "numeric patching off",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        True,
        "enumerated ablation contrast",
    ),
    ArmSpec(
        "e5-4-l2-only",
        "E5-4",
        "L2-only ladder at equal normalized work",
        "l2-only",
        10,
        100,
        3,
        4,
        3,
        True,
        "enumerated ablation contrast",
        "3a: after E2 tier iii and largest topology scale",
    ),
    ArmSpec(
        "e5-5-single-population",
        "E5-5",
        "single population",
        "ladder",
        10,
        100,
        3,
        4,
        3,
        True,
        "enumerated ablation contrast",
        "3b: after E5 contrast 4",
    ),
    ArmSpec(
        "e2-i-held-out-seeds",
        "E2-i",
        "held-out generator seeds (interpolation)",
        "l2-only",
        10,
        5,
        3,
        10,
        3,
        False,
        "generalization tier evaluation",
    ),
    ArmSpec(
        "e2-ii-network-structure",
        "E2-ii",
        "network-structure-generalization family",
        "l2-only",
        10,
        5,
        3,
        10,
        3,
        False,
        "generalization tier evaluation",
    ),
    ArmSpec(
        "e2-iii-train-ticket",
        "E2-iii",
        "unseen Train-Ticket application",
        "l2-only",
        10,
        5,
        1,
        10,
        3,
        False,
        "generalization tier evaluation",
        "1: first confirmatory-scope cut",
    ),
)


def canonical_json(value: object) -> str:
    """Return the strict repository canonical JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {"payload_sha256": canonical_sha256(payload), "payload": payload}


def _write_sealed(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    wrapped = seal_payload(payload)
    path.write_text(json.dumps(wrapped, sort_keys=True, indent=2) + "\n")
    return wrapped


def _load_sealed(path: Path, schema: str) -> dict[str, Any]:
    try:
        wrapped = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load sealed JSON {path}: {exc}") from exc
    if not isinstance(wrapped, dict) or set(wrapped) != {"payload", "payload_sha256"}:
        raise ValueError("sealed JSON must contain only payload and payload_sha256")
    payload = wrapped["payload"]
    if not isinstance(payload, dict):
        raise ValueError("sealed JSON payload must be an object")
    if wrapped["payload_sha256"] != canonical_sha256(payload):
        raise ValueError("sealed JSON payload digest mismatch")
    if payload.get("schema") != schema:
        raise ValueError(f"sealed JSON schema must be {schema!r}")
    return wrapped


def _positive_number(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number > 0")
    return float(value)


def load_ceilings(path: str | Path, *, gate_evidence: bool = False) -> dict[str, Any]:
    """Load the sealed ceilings file; missing, extra, or placeholder authority fails closed."""
    payload = _load_sealed(Path(path), CEILINGS_SCHEMA)["payload"]
    required = {
        "schema",
        "status",
        "researcher_hours_per_week",
        "weeks",
        "cpu_hours",
        "api_spend_microdollars",
        "storage_gb",
        "wallclock_days",
        "spend_authorized",
    }
    if set(payload) != required:
        raise ValueError(f"ceilings payload fields must be exactly {sorted(required)}")
    if payload["status"] not in {PLACEHOLDER_CEILINGS_STATUS, APPROVED_CEILINGS_STATUS}:
        raise ValueError("ceilings status is not recognized")
    for name in (
        "researcher_hours_per_week",
        "weeks",
        "cpu_hours",
        "storage_gb",
        "wallclock_days",
    ):
        _positive_number(name, payload[name])
    spend = payload["api_spend_microdollars"]
    if not isinstance(spend, int) or isinstance(spend, bool) or spend <= 0:
        raise ValueError("api_spend_microdollars must be a positive integer")
    if not isinstance(payload["spend_authorized"], bool):
        raise ValueError("spend_authorized must be a bool")
    if gate_evidence and payload["status"] != APPROVED_CEILINGS_STATUS:
        raise ValueError("gate evidence requires researcher-approved ceilings")
    return payload


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_quarantined_task() -> tuple[TaskEvaluationConfig, LadderConfig, dict[str, Any]]:
    digest = _sha256_file(TASK_CONFIG)
    if digest != EXPECTED_TASK_SHA256:
        raise ValueError("G0 task config differs from the quarantined 2026-07-28 pilot input")
    task = load_task_evaluation_config(TASK_CONFIG)
    ladder = load_ladder_config(TASK_CONFIG)
    actual = (
        tuple(scenario.scenario_id for scenario in task.scenarios),
        tuple(seed for scenario in task.scenarios for seed in scenario.seeds),
        task.bundle_ids,
    )
    expected = ((EXPECTED_SCENARIO,), EXPECTED_SCENARIO_SEEDS, EXPECTED_BUNDLES)
    if actual != expected:
        raise ValueError("G0 task does not contain the exact July-28 quarantined grid")
    return (
        task,
        ladder,
        {
            "task_config": str(TASK_CONFIG.relative_to(ROOT)),
            "task_config_sha256": digest,
            "data_role": "development-only",
            "pilot_binding": "experiments/pilot_codex_20260728/run_pilot.py",
            "scenarios": [EXPECTED_SCENARIO],
            "seeds": list(EXPECTED_SCENARIO_SEEDS),
            "bundles": list(EXPECTED_BUNDLES),
        },
    )


def host_snapshot() -> dict[str, Any]:
    try:
        load = os.getloadavg()
    except OSError as exc:  # pragma: no cover - supported gate hosts are Unix-like
        raise RuntimeError("G0 requires host load averages") from exc
    cpu_count = os.cpu_count()
    if not isinstance(cpu_count, int) or cpu_count < 1:
        raise RuntimeError("G0 requires a positive logical CPU count")
    return {
        "load_average_1m_5m_15m": [float(value) for value in load],
        "logical_cpu_count": cpu_count,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _l0_promotion(result: dict[str, Any], ladder: LadderConfig) -> bool:
    quality = (
        result["valid"]
        and result["feasible"]
        and result["mean_normalized_energy"] is not None
        and result["mean_normalized_energy"] <= ladder.l0.max_mean_normalized_energy
    )
    return bool(result["valid"] and (ladder.l0.mode == "validity-only" or quality))


def _l1_promotion(result: Any, ladder: LadderConfig) -> bool:
    energy_ok = result.fitness.normalized_energy is None or (
        result.fitness.normalized_energy <= ladder.l1.max_normalized_energy
    )
    return bool(
        energy_ok
        and result.fitness.summed_primitive_shortfall <= ladder.l1.max_summed_primitive_shortfall
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty timing batch")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_timing_samples(
    samples: list[dict[str, Any]], instances_per_batch: int
) -> dict[str, Any]:
    if not samples:
        raise ValueError("a rung timing batch must not be empty")
    durations = [_positive_number("duration_seconds", row["duration_seconds"]) for row in samples]
    successes = [row for row in samples if not row["failure"]]
    if not successes:
        raise ValueError("a rung timing batch with no successful call cannot price capacity")
    promotion_values = [row["promoted"] for row in successes if row["promoted"] is not None]
    return {
        "repetitions": len(samples),
        "successful_calls": len(successes),
        "p50_seconds": _percentile(durations, 0.50),
        "p90_seconds": _percentile(durations, 0.90),
        "mean_seconds": statistics.fmean(durations),
        "failure_rate": sum(bool(row["failure"]) for row in samples) / len(samples),
        "invalid_candidate_rate": sum(bool(row["invalid_candidate"]) for row in samples)
        / len(samples),
        "promotion_rate": (statistics.fmean(promotion_values) if promotion_values else None),
        "instances_per_batch": instances_per_batch,
        "p90_seconds_per_instance_linear_scaling_assumption": (
            _percentile(durations, 0.90) / instances_per_batch
        ),
        "percentile_method": "linear interpolation at (n-1)*q over all attempts",
    }


def summarize_llm_generation_samples(
    samples: list[dict[str, Any]], *, hard_call_cap: int
) -> dict[str, Any]:
    """Summarize dispatched Codex calls and fail closed on unusable pilot inputs."""
    if (
        not isinstance(hard_call_cap, int)
        or isinstance(hard_call_cap, bool)
        or hard_call_cap < 1
        or hard_call_cap > 10
    ):
        raise ValueError("LLM pilot hard call cap must be an integer in [1, 10]")
    if not samples or len(samples) > hard_call_cap:
        raise ValueError("LLM pilot samples must be non-empty and stay within the hard cap")

    token_fields = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    durations: list[float] = []
    seen_call_indices: set[int] = set()
    cli_failures = 0
    valid_bundles = 0
    invalid_bundles = 0
    l0_exercised = 0
    token_totals = {name: 0 for name in token_fields}
    for expected_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("LLM pilot samples must be objects")
        call_index = sample.get("call_index")
        if (
            not isinstance(call_index, int)
            or isinstance(call_index, bool)
            or call_index != expected_index
            or call_index in seen_call_indices
        ):
            raise ValueError("LLM pilot call indices must be contiguous and unique")
        seen_call_indices.add(call_index)
        if sample.get("dispatched") is not True:
            raise ValueError("LLM pilot samples may contain only dispatched subscription calls")
        durations.append(
            _positive_number(
                "LLM call duration_seconds", sample.get("duration_seconds")
            )
        )

        usage = sample.get("token_usage")
        if not isinstance(usage, dict) or set(usage) != set(token_fields):
            raise ValueError("every LLM call must record all four token dimensions")
        for name in token_fields:
            value = usage[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"LLM call {name} must be a non-negative integer")
            token_totals[name] += value
        if usage["cached_input_tokens"] > usage["input_tokens"]:
            raise ValueError("cached input tokens cannot exceed input tokens")

        cli_failure = sample.get("cli_failure_category") is not None
        valid_bundle = sample.get("valid_bundle")
        if not isinstance(valid_bundle, bool):
            raise ValueError("LLM call valid_bundle must be a bool")
        if cli_failure and valid_bundle:
            raise ValueError("a CLI failure cannot produce a valid bundle")
        if cli_failure:
            cli_failures += 1
        elif valid_bundle:
            valid_bundles += 1
        else:
            invalid_bundles += 1
        if sample.get("l0_exercised"):
            if not valid_bundle:
                raise ValueError("L0 cannot be exercised for an invalid bundle")
            l0_exercised += 1

    calls = len(samples)
    successful_cli_calls = calls - cli_failures
    cli_retry_factor = (
        None if successful_cli_calls == 0 else calls / successful_cli_calls
    )
    invalid_retry_factor = (
        None if valid_bundles == 0 else successful_cli_calls / valid_bundles
    )
    combined_factor = None if valid_bundles == 0 else calls / valid_bundles
    token_totals["total_tokens"] = (
        token_totals["input_tokens"] + token_totals["output_tokens"]
    )
    p90 = _percentile(durations, 0.90)
    return {
        "calls": calls,
        "successful_cli_calls": successful_cli_calls,
        "cli_failures": cli_failures,
        "valid_bundles": valid_bundles,
        "invalid_bundles": invalid_bundles,
        "l0_exercised_bundles": l0_exercised,
        "latency_seconds": {
            "p50": _percentile(durations, 0.50),
            "p90": p90,
            "mean": statistics.fmean(durations),
            "samples": durations,
            "percentile_method": "linear interpolation at (n-1)*q over all dispatched calls",
        },
        "rates": {
            "cli_failure_rate": cli_failures / calls,
            "invalid_bundle_rate_among_cli_successes": (
                None
                if successful_cli_calls == 0
                else invalid_bundles / successful_cli_calls
            ),
            "end_to_end_retry_invalid_rate": (cli_failures + invalid_bundles) / calls,
        },
        "inflation": {
            "cli_retry_factor": cli_retry_factor,
            "invalid_bundle_retry_factor": invalid_retry_factor,
            "retry_invalid_factor": combined_factor,
            "retry_invalid_fraction": (
                None if combined_factor is None else combined_factor - 1.0
            ),
        },
        "token_totals": token_totals,
        "implied_calls_per_hour_at_p90": 3600.0 / p90,
    }


def _run_timed_rung(
    rung: str,
    adapter: Any,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    seed: int,
) -> tuple[bool, bool, bool, str | None]:
    try:
        if rung == "L0":
            result = _run_l0_adapter(adapter.scorer, task)
            return False, not result["valid"], _l0_promotion(result, ladder), None
        if rung == "L1":
            result = _run_l1_adapter(
                adapter.scorer,
                adapter.proposer,
                task,
                ladder,
                seed,
                policy_name=adapter.name,
                policy_version=adapter.version,
                policy_identity_sha256=adapter.identity_sha256,
                policy_artifact_sha256=adapter.artifact_sha256,
            )
            return False, False, _l1_promotion(result, ladder), None
        if rung == "L2":
            adapter.evaluate(task, proposal_seq=seed)
            return False, False, True, None
        raise ValueError(f"unknown rung {rung}")
    except Exception as exc:  # noqa: BLE001 - failure rate is a required G0 measurement
        return True, True, False, f"{type(exc).__name__}: {exc}"


def measure_rungs(
    *,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    repetitions: dict[str, int],
    pilot_seeds: tuple[int, ...],
    evidence_class: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    if set(repetitions) != {"L0", "L1", "L2"}:
        raise ValueError("repetitions must contain exactly L0, L1, and L2")
    for rung, count in repetitions.items():
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"{rung} repetitions must be a positive int")
    if len(pilot_seeds) < max(repetitions.values()):
        raise ValueError("not enough explicit pilot seeds for the requested repetitions")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in pilot_seeds):
        raise ValueError("pilot seeds must be explicit non-negative integers")
    if len(pilot_seeds) != len(set(pilot_seeds)):
        raise ValueError("pilot seeds must be unique")

    adapters = all_handwritten()
    samples: dict[str, list[dict[str, Any]]] = {"L0": [], "L1": [], "L2": []}
    for rung in ("L0", "L1", "L2"):
        for index in range(repetitions[rung]):
            seed = pilot_seeds[index]
            adapter = adapters[index % len(adapters)]
            before = host_snapshot()
            started = time.perf_counter()
            failure, invalid, promoted, error = _run_timed_rung(rung, adapter, task, ladder, seed)
            duration = time.perf_counter() - started
            after = host_snapshot()
            samples[rung].append(
                {
                    "batch_index": index,
                    "seed": seed,
                    "policy": adapter.name,
                    "duration_seconds": duration,
                    "failure": failure,
                    "invalid_candidate": invalid,
                    "promoted": None if failure else promoted,
                    "error": error,
                    "host_before": before,
                    "host_after": after,
                    "evidence_class": evidence_class,
                }
            )

    instances = {
        "L0": len(task.scenarios) * len(task.bundle_ids),
        "L1": len(ladder.l1.scenario_ids)
        * ladder.l1.seeds_per_scenario
        * len(ladder.l1.bundle_ids),
        "L2": len(task.instance_keys) * len(task.bundle_ids),
    }
    summaries = {
        rung: summarize_timing_samples(rows, instances[rung]) for rung, rows in samples.items()
    }
    return samples, summaries


def _host_context_from_samples(samples: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    snapshots = [
        row[side]
        for rows in samples.values()
        for row in rows
        for side in ("host_before", "host_after")
    ]
    if not snapshots:
        raise ValueError("host context requires timing samples")
    static = {
        (row["logical_cpu_count"], row["platform"], row["machine"], row["python"])
        for row in snapshots
    }
    if len(static) != 1:
        raise ValueError("host identity changed during the timing pilot")
    cpu_count, platform_name, machine, python = static.pop()
    load_vectors = [row["load_average_1m_5m_15m"] for row in snapshots]
    max_normalized_any_window = max(max(row) for row in load_vectors) / cpu_count
    return {
        "logical_cpu_count": cpu_count,
        "platform": platform_name,
        "machine": machine,
        "python": python,
        "load_average_min_1m_5m_15m": [min(row[i] for row in load_vectors) for i in range(3)],
        "load_average_max_1m_5m_15m": [max(row[i] for row in load_vectors) for i in range(3)],
        "max_normalized_load_1m": max(row[0] for row in load_vectors) / cpu_count,
        "max_normalized_load_any_window": max_normalized_any_window,
        "sampling": "before and after every timed rung repetition batch",
    }


def llm_host_context(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the before/after load snapshots retained for every live CLI call."""
    return {
        **_host_context_from_samples({"LLM": samples}),
        "sampling": "immediately before and after every live Codex CLI call",
        "load_context": (
            "Development-test execution inside an active research session; permanently "
            "classified as load-contaminated regardless of observed averages."
        ),
    }


def reserve_check(usage: int | float, ceiling: int | float) -> dict[str, Any]:
    use = Decimal(str(usage))
    cap = Decimal(str(ceiling))
    if use < 0 or cap <= 0:
        raise ValueError("reserve arithmetic requires usage >= 0 and ceiling > 0")
    remaining = cap - use
    margin = remaining / cap
    return {
        "usage": float(use),
        "ceiling": float(cap),
        "remaining": float(remaining),
        "reserve_fraction": float(margin),
        "reserve_percent": float(margin * 100),
        "required_reserve_fraction": float(RESERVE_FRACTION),
        "holds": margin >= RESERVE_FRACTION,
    }


def _retry_factor(summary: dict[str, Any]) -> tuple[float, float]:
    observed = min(
        1.0,
        float(summary["failure_rate"]) + float(summary["invalid_candidate_rate"]),
    )
    priced_rate = max(observed, CAPACITY_FAILURE_RATE_FLOOR)
    if priced_rate >= 1.0:
        raise ValueError("100% failed/invalid rung cannot be capacity-priced")
    return 1.0 / (1.0 - priced_rate), priced_rate


def _rung_call_plan(
    spec: ArmSpec, timing: dict[str, dict[str, Any]], audit_fraction: float
) -> dict[str, dict[str, float]]:
    candidates = float(spec.campaigns * spec.expected_candidates_per_campaign)
    calls = {
        rung: {
            "ordinary_candidate_calls": 0.0,
            "epsilon_audit_calls": 0.0,
            "calls_before_failure_retry": 0.0,
            "calls_after_failure_retry": 0.0,
            "priced_failure_invalid_rate": 0.0,
        }
        for rung in ("L0", "L1", "L2")
    }
    if spec.workload == "l2-only":
        calls["L2"]["ordinary_candidate_calls"] = candidates
        calls["L2"]["calls_before_failure_retry"] = candidates
    else:
        p0 = float(timing["L0"]["promotion_rate"])
        p1 = float(timing["L1"]["promotion_rate"])
        calls["L0"]["ordinary_candidate_calls"] = candidates
        calls["L0"]["calls_before_failure_retry"] = candidates
        calls["L1"]["ordinary_candidate_calls"] = candidates * p0
        calls["L1"]["epsilon_audit_calls"] = candidates * (1.0 - p0) * audit_fraction
        calls["L1"]["calls_before_failure_retry"] = (
            calls["L1"]["ordinary_candidate_calls"] + calls["L1"]["epsilon_audit_calls"]
        )
        l1_entrants = calls["L1"]["calls_before_failure_retry"]
        calls["L2"]["ordinary_candidate_calls"] = l1_entrants * p1
        calls["L2"]["epsilon_audit_calls"] = l1_entrants * (1.0 - p1) * audit_fraction
        calls["L2"]["calls_before_failure_retry"] = (
            calls["L2"]["ordinary_candidate_calls"] + calls["L2"]["epsilon_audit_calls"]
        )
    for rung in ("L0", "L1", "L2"):
        factor, priced_rate = _retry_factor(timing[rung])
        calls[rung]["calls_after_failure_retry"] = (
            calls[rung]["calls_before_failure_retry"] * factor
        )
        calls[rung]["priced_failure_invalid_rate"] = priced_rate
    return calls


def _capacity_row(
    spec: ArmSpec,
    timing: dict[str, dict[str, Any]],
    audit_fraction: float,
    llm_pricing: dict[str, float],
) -> dict[str, Any]:
    calls = _rung_call_plan(spec, timing, audit_fraction)
    grid_multiplier = spec.scenarios * spec.seeds * spec.bundles
    cpu_seconds = 0.0
    storage_mb = 0.0
    for rung in ("L0", "L1", "L2"):
        invocation_count = calls[rung]["calls_after_failure_retry"]
        instance_calls = invocation_count * grid_multiplier
        calls[rung]["scenario_seed_bundle_instance_calls"] = instance_calls
        calls[rung]["p90_seconds_per_instance"] = timing[rung][
            "p90_seconds_per_instance_linear_scaling_assumption"
        ]
        calls[rung]["p90_cpu_seconds"] = instance_calls * calls[rung]["p90_seconds_per_instance"]
        cpu_seconds += calls[rung]["p90_cpu_seconds"]
        storage_mb += instance_calls * STORAGE_MB_PER_INSTANCE[rung]
    base_calls = sum(value["calls_before_failure_retry"] for value in calls.values())
    inflated_calls = sum(value["calls_after_failure_retry"] for value in calls.values())
    base_generation_proposals = (
        spec.campaigns * spec.expected_candidates_per_campaign
        if spec.generation_required
        else 0
    )
    generation_calls = (
        base_generation_proposals * llm_pricing["retry_invalid_factor"]
    )
    generation_hours = (
        generation_calls * llm_pricing["p90_seconds"] / 3600.0
    )
    researcher_hours = RESEARCHER_HOURS_PER_ARM + spec.campaigns * RESEARCHER_HOURS_PER_CAMPAIGN
    return {
        "arm_id": spec.arm_id,
        "experiment": spec.experiment,
        "method": spec.method,
        "role": spec.role,
        "workload": spec.workload,
        "campaigns": spec.campaigns,
        "campaign_count_status": "provisional",
        "campaign_count_source": (
            "P2 simulation-based precision analysis on quarantined pilot variance outstanding"
        ),
        "expected_candidates_per_campaign": spec.expected_candidates_per_campaign,
        "candidate_count_status": "provisional",
        "scenarios": spec.scenarios,
        "seeds": spec.seeds,
        "bundles": spec.bundles,
        "grid_count_status": "provisional pending P2 final matrix",
        "rung_calls": calls,
        "failure_re_evaluation_inflation": {
            "calls_before_failure_retry": base_calls,
            "calls_after_failure_retry": inflated_calls,
            "factor": inflated_calls / base_calls if base_calls else 1.0,
            "epsilon_audit_fraction": audit_fraction,
            "failure_invalid_rate_floor": CAPACITY_FAILURE_RATE_FLOOR,
        },
        "generation": {
            "required": spec.generation_required,
            "engine": LLM_ENGINE_DISPLAY if spec.generation_required else None,
            "base_proposals": base_generation_proposals,
            "calls_including_retry_invalid_inflation": generation_calls,
            "retry_invalid_inflation_factor": (
                llm_pricing["retry_invalid_factor"] if spec.generation_required else 1.0
            ),
            "retry_invalid_inflation_fraction": (
                llm_pricing["retry_invalid_fraction"] if spec.generation_required else 0.0
            ),
            "p90_seconds_per_call": (
                llm_pricing["p90_seconds"] if spec.generation_required else 0.0
            ),
            "subscription_rate_constraint": {
                "required_calls": generation_calls,
                "implied_calls_per_hour_at_p90": (
                    llm_pricing["implied_calls_per_hour_at_p90"]
                    if spec.generation_required
                    else 0.0
                ),
                "sequential_occupancy_hours": generation_hours,
                "accounting_basis": (
                    "subscription latency/rate occupancy only; no dollar or serving-energy estimate"
                    if spec.generation_required
                    else "not applicable"
                ),
            },
        },
        "resources": {
            "researcher_hours": researcher_hours,
            "cpu_hours": cpu_seconds / 3600.0,
            "storage_gb": storage_mb / 1000.0,
            "llm_generation_hours": generation_hours,
        },
        "cut_order": spec.cut_order,
    }


def _registered_gsa_line(timing: dict[str, dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    bundles = build_development_bundles(load_pinned_sources())
    strata = [stratum_for_bundle(bundle) for bundle in bundles]
    design = {
        "registration_status": "registered-design; sample counts provisional until P2",
        "n_trajectories": 20,
        "n_base": 512,
        "swap_points": 100,
        "shapley": {"d": 3, "n_outer": 32, "n_inner": 32},
        "seconds_per_eval_source": "measured L2 p90 batch runtime",
    }
    cost = estimate_compute_cost(
        strata,
        n_trajectories=design["n_trajectories"],
        n_base=design["n_base"],
        swap_points=design["swap_points"],
        seconds_per_eval=timing["L2"]["p90_seconds"],
        shapley=(
            design["shapley"]["d"],
            design["shapley"]["n_outer"],
            design["shapley"]["n_inner"],
        ),
    )
    return cost, design


def load_llm_generation_pilot(path: str | Path) -> dict[str, Any]:
    wrapped = _load_sealed(Path(path), LLM_PILOT_SCHEMA)
    payload = wrapped["payload"]
    required = {
        "schema",
        "artifact_revision",
        "status",
        "evidence_class",
        "engine_descriptor",
        "hard_call_cap",
        "configured_proposals",
        "campaign_root_seed",
        "archive",
        "summary",
        "load_context",
        "samples",
        "provenance",
    }
    if set(payload) != required:
        raise ValueError("LLM generation pilot has missing or extra fields")
    if (
        payload["status"] != "MEASURED"
        or payload["evidence_class"] != DEVELOPMENT_EVIDENCE_CLASS
    ):
        raise ValueError("Codex LLM pilot must remain load-contaminated development evidence")
    descriptor = payload["engine_descriptor"]
    if not isinstance(descriptor, dict) or (
        descriptor.get("engine"),
        descriptor.get("model_id"),
        descriptor.get("a1_reference"),
        descriptor.get("development_only"),
    ) != (CODEX_ENGINE_NAME, CODEX_MODEL_ID, A1_REFERENCE, True):
        raise ValueError("LLM generation pilot is not the A1 Codex subscription engine")
    hard_cap = payload["hard_call_cap"]
    if hard_cap != 10:
        raise ValueError("G0 Codex pilot must encode the researcher's 10-call hard cap")
    configured = payload["configured_proposals"]
    if configured != hard_cap:
        raise ValueError("G0 Codex pilot must configure exactly the hard-capped 10 proposals")
    samples = payload["samples"]
    if not isinstance(samples, list) or len(samples) != configured:
        raise ValueError("G0 Codex pilot must retain one sample per configured proposal")
    for index, sample in enumerate(samples):
        if not isinstance(sample.get("proposal_seq"), int) or sample["proposal_seq"] != index + 1:
            raise ValueError("LLM pilot proposal sequences must be contiguous after the seed")
        for name in ("selection_seed", "arm_primary_seed", "call_seed"):
            seed = sample.get(name)
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
                raise ValueError(f"LLM pilot {name} must be explicit in [0, 2**63)")
        for side in ("host_before", "host_after"):
            snapshot = sample.get(side)
            if (
                not isinstance(snapshot, dict)
                or not isinstance(snapshot.get("load_average_1m_5m_15m"), list)
                or len(snapshot["load_average_1m_5m_15m"]) != 3
                or not isinstance(snapshot.get("logical_cpu_count"), int)
                or snapshot["logical_cpu_count"] < 1
            ):
                raise ValueError("LLM generation pilot must record host load before/after")
    summary = summarize_llm_generation_samples(samples, hard_call_cap=hard_cap)
    if summary != payload["summary"]:
        raise ValueError("LLM generation pilot summary differs from its per-call samples")
    if summary["inflation"]["retry_invalid_factor"] is None:
        raise ValueError("LLM generation pilot produced no loadable bundle to capacity-price")
    if not isinstance(payload["load_context"], dict) or not payload["load_context"]:
        raise ValueError("LLM generation pilot requires a recorded load context")
    return {**payload, "sealed_payload_sha256": wrapped["payload_sha256"]}


def _resource_totals(rows: list[dict[str, Any]], gsa_cost: Any) -> dict[str, float]:
    totals = {
        key: math.fsum(row["resources"][key] for row in rows)
        for key in (
            "researcher_hours",
            "cpu_hours",
            "storage_gb",
            "llm_generation_hours",
        )
    }
    totals["researcher_hours"] += GSA_RESEARCHER_HOURS
    totals["cpu_hours"] += gsa_cost.total_hours
    totals["storage_gb"] += gsa_cost.total_evals * GSA_STORAGE_MB_PER_EVALUATION / 1000.0
    return totals


def _reserve_table(
    totals: dict[str, float], ceilings: dict[str, Any], parallel_cpu_workers: int
) -> tuple[dict[str, dict[str, Any]], float]:
    wallclock_days = (
        totals["cpu_hours"] / parallel_cpu_workers + totals["llm_generation_hours"]
    ) / 24.0
    checks = {
        "researcher_hours": reserve_check(
            totals["researcher_hours"],
            ceilings["researcher_hours_per_week"] * ceilings["weeks"],
        ),
        "cpu_hours": reserve_check(totals["cpu_hours"], ceilings["cpu_hours"]),
        "storage_gb": reserve_check(totals["storage_gb"], ceilings["storage_gb"]),
        "wallclock_days": reserve_check(wallclock_days, ceilings["wallclock_days"]),
    }
    return checks, wallclock_days


def _status_for_mode(
    mode: str, reserve_checks: dict[str, dict[str, Any]], blockers: list[str]
) -> tuple[str, dict[str, str]]:
    reserves_hold = all(check["holds"] for check in reserve_checks.values())
    if mode == "development-test":
        return (
            "DEVELOPMENT_PASS" if reserves_hold else "DEVELOPMENT_FAIL",
            {"gate_verdict": "NOT_EMITTED_FROM_DEVELOPMENT_TEST"},
        )
    if mode != "gate-evidence":
        raise ValueError("mode must be development-test or gate-evidence")
    verdict = "PASS" if reserves_hold and not blockers else "FAIL"
    return verdict, {"gate_verdict": verdict}


def build_capacity_payload(
    *,
    mode: str,
    run_id: str,
    evidence_class: str,
    timing_summary: dict[str, dict[str, Any]],
    timing_payload_sha256: str,
    host_context: dict[str, Any],
    ceilings: dict[str, Any],
    quarantine: dict[str, Any],
    audit_fraction: float,
    assumed_generation_seconds: float,
    llm_pilot: dict[str, Any] | None,
    parallel_cpu_workers: int = DEFAULT_PARALLEL_CPU_WORKERS,
    artifact_revision: str = "A1-codex-cli-r2",
) -> dict[str, Any]:
    """Build the deterministic table from fixed timing inputs and explicit assumptions."""
    if not run_id:
        raise ValueError("run_id must be explicit and non-empty")
    if evidence_class not in {DEVELOPMENT_EVIDENCE_CLASS, GATE_EVIDENCE_CLASS}:
        raise ValueError("unknown evidence class")
    if (mode == "development-test") != (evidence_class == DEVELOPMENT_EVIDENCE_CLASS):
        raise ValueError("mode and evidence class are inconsistent")
    if not isinstance(parallel_cpu_workers, int) or parallel_cpu_workers < 1:
        raise ValueError("parallel_cpu_workers must be a positive int")
    assumed_generation_seconds = _positive_number(
        "assumed_generation_seconds", assumed_generation_seconds
    )
    if not artifact_revision:
        raise ValueError("artifact_revision must be explicit and non-empty")
    if llm_pilot is None:
        llm_pricing = {
            "p90_seconds": assumed_generation_seconds,
            "retry_invalid_factor": 1.0,
            "retry_invalid_fraction": 0.0,
            "implied_calls_per_hour_at_p90": 3600.0 / assumed_generation_seconds,
        }
        generation_source = "RESEARCHER_SUPPLIED_ASSUMPTION_NOT_A_MEASUREMENT"
    else:
        summary = llm_pilot["summary"]
        factor = summary["inflation"]["retry_invalid_factor"]
        fraction = summary["inflation"]["retry_invalid_fraction"]
        if factor is None or fraction is None:
            raise ValueError("LLM pilot has no finite retry/invalid inflation factor")
        llm_pricing = {
            "p90_seconds": float(summary["latency_seconds"]["p90"]),
            "retry_invalid_factor": float(factor),
            "retry_invalid_fraction": float(fraction),
            "implied_calls_per_hour_at_p90": float(
                summary["implied_calls_per_hour_at_p90"]
            ),
        }
        generation_source = "MEASURED_CODEX_SUBSCRIPTION_P90_AND_INVALID_INFLATION"
    rows = [
        _capacity_row(
            spec,
            timing_summary,
            audit_fraction,
            llm_pricing,
        )
        for spec in PROVISIONAL_ARMS
    ]
    gsa_cost, gsa_design = _registered_gsa_line(timing_summary)
    totals = _resource_totals(rows, gsa_cost)
    reserve_checks, wallclock_days = _reserve_table(totals, ceilings, parallel_cpu_workers)
    totals["wallclock_days"] = wallclock_days

    generation_rows = [row for row in rows if row["generation"]["required"]]
    base_proposals_total = sum(
        row["generation"]["base_proposals"] for row in generation_rows
    )
    calls_total = math.fsum(
        row["generation"]["calls_including_retry_invalid_inflation"]
        for row in generation_rows
    )
    generation_hours_total = math.fsum(
        row["generation"]["subscription_rate_constraint"]["sequential_occupancy_hours"]
        for row in generation_rows
    )
    llm_line = {
        "status": "MEASURED_DEVELOPMENT_ONLY" if llm_pilot is not None else LLM_PILOT_MISSING,
        "engine": LLM_ENGINE_DISPLAY,
        "engine_id": CODEX_ENGINE_NAME,
        "model_id": CODEX_MODEL_ID,
        "a1_reference": A1_REFERENCE,
        "generation_time_source": generation_source,
        "pilot_evidence_class": (
            None if llm_pilot is None else llm_pilot["evidence_class"]
        ),
        "pilot_artifact_revision": (
            None if llm_pilot is None else llm_pilot["artifact_revision"]
        ),
        "pilot_payload_sha256": (
            None if llm_pilot is None else llm_pilot.get("sealed_payload_sha256")
        ),
        "pilot_archive": None if llm_pilot is None else llm_pilot.get("archive"),
        "pilot_load_context": (
            None if llm_pilot is None else llm_pilot.get("load_context")
        ),
        "pilot_summary": None if llm_pilot is None else llm_pilot["summary"],
        "capacity_formula": (
            "sum(campaigns * proposals_per_campaign for each generation arm) * "
            "(1 + measured retry_invalid_inflation_fraction) * measured p90 latency"
        ),
        "proposal_arm_count": len(generation_rows),
        "base_proposals_total": base_proposals_total,
        "retry_invalid_inflation_factor": llm_pricing["retry_invalid_factor"],
        "retry_invalid_inflation_fraction": llm_pricing["retry_invalid_fraction"],
        "capacity_priced_p90_seconds_per_call": llm_pricing["p90_seconds"],
        "capacity_call_count_total": calls_total,
        "subscription_rate_constraint": {
            "implied_calls_per_hour_at_p90": llm_pricing[
                "implied_calls_per_hour_at_p90"
            ],
            "sequential_occupancy_hours": generation_hours_total,
            "sequential_occupancy_days": generation_hours_total / 24.0,
            "accounting_basis": (
                "subscription rate-constraint occupancy only; A1 forbids monetary and "
                "serving-energy estimates for this engine"
            ),
        },
        "blocking_gate_pass": True,
        "blocking_reason": LLM_BLOCKING_REASON,
    }
    blockers = []
    if evidence_class == DEVELOPMENT_EVIDENCE_CLASS:
        blockers.extend(
            [
                "DEVELOPMENT_TEST_LOAD_CONTAMINATED",
                "DEVELOPMENT_TEST_CANNOT_EMIT_G0_GATE_VERDICT",
            ]
        )
    if ceilings["status"] == PLACEHOLDER_CEILINGS_STATUS:
        blockers.append("PLACEHOLDER_CEILINGS_REQUIRE_RESEARCHER_VALUES")
    if llm_pilot is None:
        blockers.append(LLM_PILOT_MISSING)
    blockers.append(LLM_BLOCKING_REASON)
    blockers.extend(
        f"CEILING_RESERVE_BELOW_30_PERCENT:{name}"
        for name, check in reserve_checks.items()
        if not check["holds"]
    )
    status, authority = _status_for_mode(mode, reserve_checks, blockers)

    payload = {
        "schema": CAPACITY_SCHEMA,
        "artifact_revision": artifact_revision,
        "run_id": run_id,
        "mode": mode,
        "evidence_class": evidence_class,
        "status": status,
        "acceptance_evidence": mode == "gate-evidence",
        **authority,
        "plan_reference": "ContinuumPlacementScopedDown.md v2.5 section 8 / T7",
        "timing_pilot_payload_sha256": timing_payload_sha256,
        "host_context": host_context,
        "quarantined_pilot_inputs": quarantine,
        "rung_timings": timing_summary,
        "llm_generation": llm_line,
        "arm_enumeration": {
            "status": "provisional",
            "freeze_rule": (
                "P2 fixes campaign/episode counts by simulation-based precision analysis "
                "on quarantined pilot variance, then recertifies this table"
            ),
            "rows": rows,
        },
        "gsa_line_item": {
            **dataclasses.asdict(gsa_cost),
            "registered_design": gsa_design,
            "as_g0_line_item": gsa_cost.as_g0_line_item(),
            "researcher_hours": GSA_RESEARCHER_HOURS,
            "storage_gb": (gsa_cost.total_evals * GSA_STORAGE_MB_PER_EVALUATION / 1000.0),
        },
        "capacity_assumptions": {
            "failure_invalid_rate_floor": CAPACITY_FAILURE_RATE_FLOOR,
            "llm_retry_invalid_inflation_source": (
                "measured Codex pilot" if llm_pilot is not None else "unmeasured fallback"
            ),
            "storage_mb_per_scenario_seed_bundle_instance": STORAGE_MB_PER_INSTANCE,
            "researcher_hours_per_arm": RESEARCHER_HOURS_PER_ARM,
            "researcher_hours_per_campaign": RESEARCHER_HOURS_PER_CAMPAIGN,
            "parallel_cpu_workers": parallel_cpu_workers,
            "wallclock_formula": (
                "(total CPU hours / provisional parallel workers + sequential LLM hours) / 24"
            ),
            "rung_scaling": (
                "pilot p90 batch time divided by pilot instance count, then multiplied by "
                "each provisional arm's literal scenario x seed x bundle count"
            ),
        },
        "ceilings": {
            "source_schema": ceilings["schema"],
            "status": ceilings["status"],
            "researcher_hours_per_week": ceilings["researcher_hours_per_week"],
            "weeks": ceilings["weeks"],
            "cpu_hours": ceilings["cpu_hours"],
            "storage_gb": ceilings["storage_gb"],
            "wallclock_days": ceilings["wallclock_days"],
            "a1_subscription_constraint": (
                "Generation is accounted as implied calls/hour and sequential occupancy; "
                "the superseded hosted-model spend ceiling is not applied."
            ),
        },
        "resource_totals": totals,
        "reserve_checks": reserve_checks,
        "all_ceilings_hold_with_30_percent_reserve": all(
            check["holds"] for check in reserve_checks.values()
        ),
        "capacity_screen": (
            "FITS_WITH_30_PERCENT_RESERVE"
            if all(check["holds"] for check in reserve_checks.values())
            else "DOES_NOT_FIT_WITH_30_PERCENT_RESERVE"
        ),
        "blocking_items": sorted(set(blockers)),
        "scope_cut_order": [
            "E2 tier (iii) unseen application",
            "largest confirmatory topology scale",
            "E5 contrast (4), then E5 contrast (5)",
            "campaign counts down to the P2 precision-analysis floor",
            "E6 modeled-lifecycle depth; measured cycle-cost table retained",
        ],
        "never_descoped": [
            "same-DSL GP ablation (G-ablate)",
            "DES verification suite",
            "G2",
            "G3-cal",
            "two-stage preregistration",
            "held-out bundle robustness",
            "public/private split",
        ],
    }
    if mode == "development-test":
        # A development artifact must never contain a PASS/FAIL gate verdict.
        payload.pop("gate_verdict", None)
        payload["gate_verdict_policy"] = "NOT_EMITTED_FROM_DEVELOPMENT_TEST"
    return payload


def _fmt(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.4f}"


def render_markdown(payload: dict[str, Any], payload_sha256: str) -> str:
    lines = [
        "# G0 throughput and capacity table",
        "",
        f"- Status: **{payload['status']}**",
        f"- Evidence class: **{payload['evidence_class']}**",
        f"- Capacity payload SHA-256: `{payload_sha256}`",
        f"- Capacity screen: **{payload['capacity_screen']}**",
        f"- Gate verdict: **{payload.get('gate_verdict', 'NOT EMITTED')}**",
        "- Counts are provisional until the P2 precision analysis and mandatory recertification.",
        "",
        "## Host and measured rung timings",
        "",
        (
            f"Host: {payload['host_context']['platform']} / "
            f"{payload['host_context']['logical_cpu_count']} logical CPUs. "
            f"Load averages observed (min to max): "
            f"{payload['host_context']['load_average_min_1m_5m_15m']} to "
            f"{payload['host_context']['load_average_max_1m_5m_15m']}."
        ),
        "",
        "| Rung | Repetitions | p50 s | p90 s | mean s | failure | invalid | promotion |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rung in ("L0", "L1", "L2"):
        row = payload["rung_timings"][rung]
        promotion = row["promotion_rate"]
        lines.append(
            f"| {rung} | {row['repetitions']} | {_fmt(row['p50_seconds'])} | "
            f"{_fmt(row['p90_seconds'])} | {_fmt(row['mean_seconds'])} | "
            f"{row['failure_rate']:.1%} | {row['invalid_candidate_rate']:.1%} | "
            f"{'n/a' if promotion is None else f'{promotion:.1%}'} |"
        )
    llm = payload["llm_generation"]
    pilot = llm["pilot_summary"]
    if pilot is None:
        pilot_text = "No sealed live pilot was supplied; the displayed timing is an assumption."
    else:
        tokens = pilot["token_totals"]
        pilot_text = (
            f"Pilot: {pilot['calls']} calls, p50={pilot['latency_seconds']['p50']:.3f} s, "
            f"p90={pilot['latency_seconds']['p90']:.3f} s, CLI failures="
            f"{pilot['rates']['cli_failure_rate']:.1%}, invalid bundles among successful "
            f"CLI calls={pilot['rates']['invalid_bundle_rate_among_cli_successes']:.1%}. "
            f"Tokens (input/cached/output/reasoning): {tokens['input_tokens']}/"
            f"{tokens['cached_input_tokens']}/{tokens['output_tokens']}/"
            f"{tokens['reasoning_output_tokens']}."
        )
    lines.extend(
        [
            "",
            "## LLM generation",
            "",
            f"Engine: **{llm['engine']}**. Status: **{llm['status']}**. {pilot_text}",
            (
                f"Capacity prices {llm['base_proposals_total']:,} base proposals as "
                f"{llm['capacity_call_count_total']:,.1f} subscription calls at "
                f"{llm['capacity_priced_p90_seconds_per_call']:.3f} s/call "
                f"({llm['subscription_rate_constraint']['implied_calls_per_hour_at_p90']:.2f} "
                "calls/hour implied). This is rate-constraint occupancy only; A1 provides "
                "neither a monetary nor serving-energy figure."
            ),
            (
                "This measured development line still blocks a gate PASS: "
                f"`{llm['blocking_reason']}`."
            ),
            "",
            "## Literal arm-level table",
            "",
            (
                "Calls include epsilon audits and a failure/invalid retry rate of at least "
                f"{payload['capacity_assumptions']['failure_invalid_rate_floor']:.1%}. "
                "The repeated measured p90 batch times are "
                f"L0={payload['rung_timings']['L0']['p90_seconds']:.4f} s, "
                f"L1={payload['rung_timings']['L1']['p90_seconds']:.4f} s, and "
                f"L2={payload['rung_timings']['L2']['p90_seconds']:.4f} s."
            ),
            "",
            "| Arm | Campaigns | Candidates/campaign | Scenarios x seeds x bundles | "
            "L0/L1/L2 inflated calls | Inflation | CPU h | Subscription occupancy | Storage GB | "
            "Researcher h |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["arm_enumeration"]["rows"]:
        calls = row["rung_calls"]
        resources = row["resources"]
        lines.append(
            f"| {row['arm_id']} | {row['campaigns']}* | "
            f"{row['expected_candidates_per_campaign']}* | "
            f"{row['scenarios']} x {row['seeds']} x {row['bundles']}* | "
            f"{calls['L0']['calls_after_failure_retry']:.1f} / "
            f"{calls['L1']['calls_after_failure_retry']:.1f} / "
            f"{calls['L2']['calls_after_failure_retry']:.1f} | "
            f"{row['failure_re_evaluation_inflation']['factor']:.3f}x | "
            f"{resources['cpu_hours']:.3f} | "
            f"{row['generation']['subscription_rate_constraint']['required_calls']:.1f} calls / "
            f"{resources['llm_generation_hours']:.3f} h | "
            f"{resources['storage_gb']:.3f} | "
            f"{resources['researcher_hours']:.2f} |"
        )
    lines.extend(
        [
            "",
            "`*` Provisional input; the real count comes from the P2 precision analysis.",
            "",
            "## Registered GSA line item",
            "",
            payload["gsa_line_item"]["as_g0_line_item"],
            "",
            "## Per-resource ceilings and reserve",
            "",
            "| Resource | Usage | Ceiling | Remaining | Reserve | >=30% |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, check in payload["reserve_checks"].items():
        lines.append(
            f"| {name} | {_fmt(check['usage'])} | {_fmt(check['ceiling'])} | "
            f"{_fmt(check['remaining'])} | {check['reserve_percent']:.2f}% | "
            f"{'yes' if check['holds'] else 'NO'} |"
        )
    lines.extend(["", "## Blocking items", ""])
    if payload["blocking_items"]:
        lines.extend(f"- {item}" for item in payload["blocking_items"])
    else:
        lines.append("- None")
    lines.extend(["", "## Pre-listed scope cut order", ""])
    lines.extend(
        f"{index}. {item}" for index, item in enumerate(payload["scope_cut_order"], start=1)
    )
    lines.extend(
        [
            "",
            "The same-DSL GP ablation and the other plan §8 non-descopable items are retained.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_seeds(raw: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in raw.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pilot seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one pilot seed is required")
    return seeds


def _gate_preflight(
    initial_host: dict[str, Any], repetitions: dict[str, int], llm_pilot: dict[str, Any] | None
) -> None:
    if llm_pilot is None:
        raise ValueError("gate evidence requires --llm-generation-pilot")
    for rung, minimum in MIN_GATE_REPETITIONS.items():
        if repetitions[rung] < minimum:
            raise ValueError(f"gate evidence requires at least {minimum} {rung} repetitions")
    normalized_load = (
        max(initial_host["load_average_1m_5m_15m"]) / initial_host["logical_cpu_count"]
    )
    if normalized_load > MAX_GATE_NORMALIZED_LOAD_1M:
        raise ValueError(
            "gate evidence refused: a normalized host-load window exceeds quiet-host ceiling"
        )


def run_harness(
    *,
    mode: str,
    output: Path,
    ceilings_path: Path,
    run_id: str,
    repetitions: dict[str, int],
    pilot_seeds: tuple[int, ...],
    assumed_generation_seconds: float,
    llm_generation_pilot_path: Path | None = None,
    parallel_cpu_workers: int = DEFAULT_PARALLEL_CPU_WORKERS,
) -> dict[str, Any]:
    if mode not in {"development-test", "gate-evidence"}:
        raise ValueError("mode must be development-test or gate-evidence")
    evidence_class = (
        DEVELOPMENT_EVIDENCE_CLASS if mode == "development-test" else GATE_EVIDENCE_CLASS
    )
    ceilings = load_ceilings(ceilings_path, gate_evidence=mode == "gate-evidence")
    llm_pilot = (
        None
        if llm_generation_pilot_path is None
        else load_llm_generation_pilot(llm_generation_pilot_path)
    )
    initial_host = host_snapshot()
    if mode == "gate-evidence":
        _gate_preflight(initial_host, repetitions, llm_pilot)
    task, ladder, quarantine = _load_quarantined_task()
    samples, timing_summary = measure_rungs(
        task=task,
        ladder=ladder,
        repetitions=repetitions,
        pilot_seeds=pilot_seeds,
        evidence_class=evidence_class,
    )
    host_context = _host_context_from_samples(samples)
    if mode == "development-test":
        host_context["operator_declared_condition"] = (
            "Host reported under heavy load (load averages typically 20-40); exact direct "
            "samples above govern this artifact"
        )
    if (
        mode == "gate-evidence"
        and host_context["max_normalized_load_any_window"] > MAX_GATE_NORMALIZED_LOAD_1M
    ):
        raise ValueError("gate evidence refused: host stopped being quiet during the timing pilot")
    deviations = [
        f"{rung} repetitions scaled below gate default ({count} < {MIN_GATE_REPETITIONS[rung]})"
        for rung, count in repetitions.items()
        if count < MIN_GATE_REPETITIONS[rung]
    ]
    timing_payload = {
        "schema": TIMING_SCHEMA,
        "run_id": run_id,
        "mode": mode,
        "evidence_class": evidence_class,
        "acceptance_evidence": mode == "gate-evidence",
        "host_context": host_context,
        "quarantined_pilot_inputs": quarantine,
        "repetitions": repetitions,
        "pilot_seeds": list(pilot_seeds[: max(repetitions.values())]),
        "candidate_payloads": [adapter.name for adapter in all_handwritten()],
        "driver": {
            "module": "evolution.g0",
            "rung_authority": "evolution.ladder Phase 5.4 real adapter entry points",
            "git_head": _git_head(),
            "files": {
                "g0_sha256": _sha256_file(Path(__file__)),
                "ladder_sha256": _sha256_file(ROOT / "evolution/ladder.py"),
                "evaluation_sha256": _sha256_file(ROOT / "evolution/evaluation.py"),
                "campaign_config_sha256": _sha256_file(CAMPAIGN_CONFIG),
            },
        },
        "runtime_scaling_deviations": deviations,
        "summaries": timing_summary,
        "samples": samples,
    }
    timing_wrapped = seal_payload(timing_payload)
    capacity_payload = build_capacity_payload(
        mode=mode,
        run_id=run_id,
        evidence_class=evidence_class,
        timing_summary=timing_summary,
        timing_payload_sha256=timing_wrapped["payload_sha256"],
        host_context=host_context,
        ceilings=ceilings,
        quarantine=quarantine,
        audit_fraction=ladder.epsilon_audit_fraction,
        assumed_generation_seconds=assumed_generation_seconds,
        llm_pilot=llm_pilot,
        parallel_cpu_workers=parallel_cpu_workers,
    )
    capacity_wrapped = seal_payload(capacity_payload)

    output.mkdir(parents=True, exist_ok=True)
    timing_path = output / "g0_rung_timing_pilot.json"
    capacity_path = output / "g0_capacity_table.json"
    markdown_path = output / "G0_CAPACITY_TABLE.md"
    for path in (timing_path, capacity_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite G0 artifact: {path}")
    _write_sealed(timing_path, timing_payload)
    _write_sealed(capacity_path, capacity_payload)
    markdown_path.write_text(render_markdown(capacity_payload, capacity_wrapped["payload_sha256"]))
    return capacity_wrapped


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development-test", "gate-evidence"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ceilings", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--l0-repetitions", type=int, default=30)
    parser.add_argument("--l1-repetitions", type=int, default=10)
    parser.add_argument("--l2-repetitions", type=int, default=5)
    parser.add_argument(
        "--pilot-seeds",
        type=_parse_seeds,
        default=DEFAULT_PILOT_SEEDS,
        help="explicit comma-separated repetition seeds",
    )
    parser.add_argument("--assumed-llm-generation-seconds", type=float, required=True)
    parser.add_argument("--llm-generation-pilot", type=Path)
    parser.add_argument("--parallel-cpu-workers", type=int, default=DEFAULT_PARALLEL_CPU_WORKERS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    wrapped = run_harness(
        mode=args.mode,
        output=args.output,
        ceilings_path=args.ceilings,
        run_id=args.run_id,
        repetitions={
            "L0": args.l0_repetitions,
            "L1": args.l1_repetitions,
            "L2": args.l2_repetitions,
        },
        pilot_seeds=args.pilot_seeds,
        assumed_generation_seconds=args.assumed_llm_generation_seconds,
        llm_generation_pilot_path=args.llm_generation_pilot,
        parallel_cpu_workers=args.parallel_cpu_workers,
    )
    payload = wrapped["payload"]
    print(
        canonical_json(
            {
                "status": payload["status"],
                "capacity_screen": payload["capacity_screen"],
                "payload_sha256": wrapped["payload_sha256"],
                "output": str(args.output),
            }
        )
    )
    if args.mode == "development-test":
        return 0 if payload["status"] == "DEVELOPMENT_PASS" else 1
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
