"""Fail-closed engine-path preflight with A1 and frozen-PPO verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from baselines.ppo import load_frozen_ppo_bundle
from baselines.same_dsl_gp import SameDSLGPArm
from evolution.campaign import load_campaign_plan
from evolution.codex_engine import (
    CODEX_ENGINE_NAME,
    CodexCLICompletionClient,
    load_codex_engine_config,
    refuse_codex_confirmatory_engine,
    validate_codex_development_profile,
)
from evolution.evaluation import load_task_evaluation_config
from evolution.feedback import load_feedback_profiles
from evolution.islands import load_islands_config
from evolution.ladder import load_ladder_config
from evolution.model import load_model_track
from evolution.openrouter_engine import (
    OpenRouterCompletionClient,
    load_openrouter_engine_config,
)
from evolution.profiles import MATCHED_PROFILE

from .support import (
    BOUNDARY,
    MODEL_MANIFEST,
    ROOT,
    PreparedRun,
    canonical_json,
    file_sha256,
    required_call_ceiling,
    write_sealed,
)

ENGINE_PROBE_MARKER = "EVOLVED_CONTINUUM_CODEX_ENGINE_PREFLIGHT_OK"
ENGINE_PROBE_SEED = 2_026_082_101
OPENROUTER_PROBE_MARKER = "EVOLVED_CONTINUUM_OPENROUTER_ENGINE_PREFLIGHT_OK"


def run_engine_path_probe(
    client: CodexCLICompletionClient, codex_config: Any
) -> dict[str, Any]:
    completion = client.complete(
        model=codex_config.model_id,
        prompt=f"Reply with exactly {ENGINE_PROBE_MARKER} and no other text.",
        generation=codex_config.generation,
        seed=ENGINE_PROBE_SEED,
    )
    if completion.content.strip() != ENGINE_PROBE_MARKER:
        raise RuntimeError(
            "Codex engine-path preflight returned a parsed completion but not "
            f"the exact marker {ENGINE_PROBE_MARKER!r}"
        )
    raw = completion.raw_response
    attempt = raw["attempts"][0]
    return {
        "status": "PARSED_COMPLETION",
        "seed": ENGINE_PROBE_SEED,
        "marker": ENGINE_PROBE_MARKER,
        "session_id": raw["session_id"],
        "token_usage": raw["token_usage"],
        "duration_ms": attempt["process"]["duration_ms"],
        "argv": attempt["process"]["argv"],
        "runtime_state": attempt["process"]["runtime_state"],
    }


def run_openrouter_engine_path_probe(
    client: OpenRouterCompletionClient, openrouter_config: Any
) -> dict[str, Any]:
    completion = client.complete(
        model=openrouter_config.model_id,
        prompt=f"Reply with exactly {OPENROUTER_PROBE_MARKER} and no other text.",
        generation=openrouter_config.generation,
        seed=ENGINE_PROBE_SEED,
    )
    if completion.content.strip() != OPENROUTER_PROBE_MARKER:
        raise RuntimeError(
            "OpenRouter engine-path preflight returned a completion but not "
            f"the exact marker {OPENROUTER_PROBE_MARKER!r}"
        )
    return {
        "status": "PARSED_COMPLETION",
        "seed": ENGINE_PROBE_SEED,
        "marker": OPENROUTER_PROBE_MARKER,
        "model_id": completion.raw_response["model"],
        "token_usage": completion.raw_response["usage"],
        "cost_microusd": completion.cost_microusd,
    }


def verify_a1_refusals(codex_config: Any, *, feedback_profile: str) -> dict[str, bool]:
    validate_codex_development_profile(
        codex_config,
        development=True,
        campaign_status="development-provisional",
        feedback_profile=feedback_profile,
    )
    development_flag_refused = False
    try:
        validate_codex_development_profile(
            codex_config,
            development=False,
            campaign_status="development-provisional",
            feedback_profile=feedback_profile,
        )
    except ValueError:
        development_flag_refused = True
    confirmatory_boundary_refused = False
    try:
        refuse_codex_confirmatory_engine(
            {"engine": CODEX_ENGINE_NAME},
            boundary="all-baselines harness preflight confirmatory probe",
        )
    except ValueError:
        confirmatory_boundary_refused = True
    if not development_flag_refused or not confirmatory_boundary_refused:
        raise RuntimeError("A1 development-only Codex refusals are not active")
    return {
        "missing_development_override_refused": development_flag_refused,
        "confirmatory_boundary_refused": confirmatory_boundary_refused,
    }


def run_preflight(prepared: PreparedRun, *, inspect_cli: bool = True) -> dict[str, Any]:
    plan = load_campaign_plan(prepared.campaign_config)
    task = load_task_evaluation_config(prepared.task_config)
    ladder = load_ladder_config(prepared.task_config)
    islands = load_islands_config(prepared.task_config)
    profiles = load_feedback_profiles(prepared.task_config)
    load_model_track(MODEL_MANIFEST)
    codex = None
    openrouter = None
    if prepared.engine == "codex":
        codex = load_codex_engine_config(prepared.engine_config)
    else:
        openrouter = load_openrouter_engine_config(prepared.engine_config)
    definition = plan.campaigns[0]
    config = prepared.harness_config
    if (
        definition.campaign_id != "h00"
        or definition.root_seed != prepared.root_seed
        or definition.evaluation_replicates != config["evaluation_replicates"]
    ):
        raise ValueError("generated campaign definition differs from the requested run")
    if (
        plan.status != "development-provisional"
        or plan.evaluator_work.registration_status != "development-provisional"
        or plan.evaluator_work.max_microwork != prepared.max_microwork
        or plan.evaluator_work.rung_microwork != config["rung_microwork"]
        or plan.evolution_profile != prepared.profile_name
        or plan.ablation_probe_probability
        != prepared.ablation_probe_probability
        or plan.ablation_prompt_hint_probability
        != prepared.ablation_prompt_hint_probability
        or plan.proposals_per_call != prepared.proposals_per_call
        or plan.feedback_profile != prepared.evolution_profile.feedback_profile
        or profiles[plan.feedback_profile].enabled
        is not prepared.evolution_profile.simulator_feedback_enabled
        or ladder.registration_status != "development-provisional"
        or islands.latency_calibration.status != "development-provisional"
    ):
        raise ValueError("generated run differs from its development evolution profile")
    if len(task.scenarios) != 1 or task.scenarios[0].scenario_id != prepared.scenario_id:
        raise ValueError("runtime task does not contain exactly the named scenario")
    if any(not bundle.startswith("dev-") for bundle in task.bundle_ids):
        raise ValueError("runtime task crossed the development bundle boundary")
    if codex is not None:
        if (
            codex.engine != config["llm"]["engine"]
            or codex.max_calls_per_campaign
            != required_call_ceiling(config, prepared.max_microwork)
            or codex.max_wall_clock_s_per_campaign
            != config["llm"]["hard_wall_clock_seconds"]
        ):
            raise ValueError("LLM hard ceilings differ between harness and engine config")
        refusals: dict[str, bool] | None = verify_a1_refusals(
            codex, feedback_profile=plan.feedback_profile
        )
    else:
        if prepared.engine != "openrouter" or openrouter is None:
            raise ValueError("selected engine differs from its config")
        refusals = None
    same_dsl = SameDSLGPArm()
    if prepared.gp_config_path is not None:
        from evolution.gp_operator import load_gp_operator_config

        gp_config, _label = load_gp_operator_config(prepared.gp_config_path)
        same_dsl = SameDSLGPArm(config=gp_config)
    if plan.evolution_profile == MATCHED_PROFILE:
        same_dsl.validate_matched_profile(plan, profiles[plan.feedback_profile])

    ppo_dir = (ROOT / config["ppo"]["artifact_dir"]).resolve()
    expected_ppo_receipt = config["ppo"]["expected_training_receipt_sha256"]
    ppo_bundle = load_frozen_ppo_bundle(
        ppo_dir,
        expected_receipt_sha256=expected_ppo_receipt,
        require_p2_frozen=False,
    )
    if ppo_bundle.metadata.get("status") != "development-provisional":
        raise ValueError("harness PPO reference is not development-provisional")

    seed_archive = None
    if prepared.population_parity:
        from .initialization import read_seed_archive

        path = prepared.run_dir / "inputs/seed_archive.json"
        seed_archive = read_seed_archive(
            path,
            expected_size=prepared.initialization_population_size,
            initial_path=prepared.initial_path,
        )

    cli_version = None
    engine_probe = None
    if inspect_cli:
        if codex is not None:
            client = CodexCLICompletionClient(codex)
            if (
                client.usage_ledger.max_calls
                != required_call_ceiling(config, prepared.max_microwork)
                or client.usage_ledger.max_wall_clock_ms
                != int(config["llm"]["hard_wall_clock_seconds"] * 1_000)
            ):
                raise RuntimeError("Codex runner-facing usage ledger has different ceilings")
            cli_version = client.cli_version
            if prepared.include_llm:
                engine_probe = run_engine_path_probe(client, codex)
        elif prepared.include_llm:
            engine_probe = run_openrouter_engine_path_probe(
                OpenRouterCompletionClient(openrouter), openrouter
            )
    selected_config = codex if codex is not None else openrouter
    assert selected_config is not None
    payload = {
        "schema": "evolved-continuum-allbaselines-preflight-v1",
        "status": (
            "READY_ENGINE_PROBE_PASSED"
            if prepared.include_llm and inspect_cli
            else "READY_NO_LLM_ENGINE_PROBE_REQUIRED"
        ),
        "development_only": True,
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "scenario_id": prepared.scenario_id,
        "root_seed": prepared.root_seed,
        "include_llm": prepared.include_llm,
        "evolution_profile": prepared.profile_name,
        "matched_max_microwork": prepared.max_microwork,
        "initialization": {
            "mode": (
                "population-parity" if prepared.population_parity else "single-seed"
            ),
            "population_size": prepared.initialization_population_size,
            "seed_archive_random_mutants": prepared.seed_archive_random_mutants,
            "initial_source_path": str(prepared.initial_path.relative_to(ROOT)),
            "seed_archive_payload_sha256": (
                None if seed_archive is None else seed_archive["payload_sha256"]
            ),
            "charging_rule": "EvaluatorWorkBudget.work_for_trace",
            "gphh_generation0_population_size": config[
                "classic_gphh_population_size"
            ],
        },
        "rung_microwork": plan.evaluator_work.rung_microwork,
        "evaluation_replicates": definition.evaluation_replicates,
        "llm_slot": {
            "engine": prepared.engine if codex is None else codex.engine,
            "model_id": selected_config.model_id,
            "enabled_for_this_run": prepared.include_llm,
            "proposals_per_call": prepared.proposals_per_call,
            "cli_version": cli_version,
            "completion_calls_made_by_preflight": (
                1 if prepared.include_llm and inspect_cli else 0
            ),
            "engine_path_probe": engine_probe,
        },
        "same_dsl_gp": same_dsl.receipt(),
        "ppo_frozen": {
            "artifact_sha256": ppo_bundle.artifact_sha256,
            "training_receipt_sha256": expected_ppo_receipt,
            "adaptation_budget": None,
        },
        "inputs": {
            "campaign_config_sha256": file_sha256(prepared.campaign_config),
            "task_config_sha256": file_sha256(prepared.task_config),
            "engine_config_sha256": file_sha256(prepared.engine_config),
            "model_manifest_sha256": file_sha256(MODEL_MANIFEST),
        },
    }
    if prepared.gp_config_path is not None:
        payload["gp_mix_label"] = prepared.gp_mix_label
        payload["inputs"]["gp_operator_config_sha256"] = file_sha256(
            prepared.gp_config_path
        )
    if codex is not None:
        payload["llm_slot"].update(
            {
                "hard_call_ceiling": codex.max_calls_per_campaign,
                "hard_wall_clock_seconds": codex.max_wall_clock_s_per_campaign,
                "startup_grace_seconds": codex.startup_grace_s,
                "max_consecutive_failures": codex.max_consecutive_failures,
                "journal_watchdog_multiplier": codex.journal_watchdog_multiplier,
            }
        )
        payload["a1_refusals"] = refusals
        payload["inputs"]["codex_engine_config_sha256"] = payload["inputs"].pop(
            "engine_config_sha256"
        )
    else:
        payload["llm_slot"].update(
            {
                "config_sha256": file_sha256(prepared.engine_config),
                "resolved_price_table": {
                    "prompt_microusd_per_million": openrouter.prompt_microusd_per_million,
                    "completion_microusd_per_million": (
                        openrouter.completion_microusd_per_million
                    ),
                },
            }
        )
    if (
        prepared.ablation_probe_probability > 0.0
        or prepared.ablation_prompt_hint_probability > 0.0
    ):
        payload["ablation_controls"] = {
            "ablation_probe_probability": prepared.ablation_probe_probability,
            "ablation_prompt_hint_probability": (
                prepared.ablation_prompt_hint_probability
            ),
        }
    # A run that is interrupted must be resumable. The engine-path probe is a *live*
    # dispatch, so its session id, duration and token counts differ on every execution;
    # sealing it into `preflight.json` made `write_exact` refuse the rewrite on resume
    # and turned every interruption into a discarded run. The probe is evidence that the
    # engine path worked, not an input the run is reproducible against, so it is written
    # once to its own append-only record and excluded from the sealed payload.
    slot = payload.get("llm_slot")
    probe = slot.pop("engine_path_probe", None) if isinstance(slot, dict) else None
    if probe is not None:
        journal = prepared.run_dir / "engine-path-probes.jsonl"
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(probe) + "\n")
        slot["engine_path_probe"] = {
            "status": probe.get("status"),
            "marker": probe.get("marker"),
            "seed": probe.get("seed"),
            "record": "engine-path-probes.jsonl",
        }
    write_sealed(prepared.run_dir / "preflight.json", payload)
    return payload


def preflight_from_paths(
    scenario_id: str,
    task_config: Path,
    normalized_budget: str,
    root_seed: int,
    *,
    include_llm: bool,
    profile_name: str = MATCHED_PROFILE,
    ablation_probe_probability: float = 0.0,
    ablation_prompt_hint_probability: float = 0.0,
    proposals_per_call: int = 1,
    seed_archive_random_mutants: int | None = None,
    initial_path: object | None = None,
    runs_root: str | Path | None = None,
    engine: str = "codex",
    engine_config: str | Path | None = None,
    gp_config_path: str | Path | None = None,
    inspect_cli: bool = True,
) -> tuple[PreparedRun, dict[str, Any]]:
    from .support import prepare_run

    prepared = prepare_run(
        scenario_id,
        task_config,
        normalized_budget,
        root_seed,
        include_llm=include_llm,
        profile_name=profile_name,
        ablation_probe_probability=ablation_probe_probability,
        ablation_prompt_hint_probability=ablation_prompt_hint_probability,
        proposals_per_call=proposals_per_call,
        seed_archive_random_mutants=seed_archive_random_mutants,
        initial_path=initial_path,
        runs_root=runs_root,
        engine=engine,
        engine_config=engine_config,
        gp_config_path=gp_config_path,
    )
    if prepared.population_parity:
        from .initialization import materialize_seed_archive

        materialize_seed_archive(prepared)
    return prepared, run_preflight(prepared, inspect_cli=inspect_cli)


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))
