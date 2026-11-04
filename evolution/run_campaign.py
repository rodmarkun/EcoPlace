#!/usr/bin/env python3
"""CLI for one resumable Phase-5 campaign arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from evolution.campaign import (
    CampaignRunner,
    load_campaign_plan,
    verify_confirmatory_authorization,
)
from evolution.codex_engine import (
    CODEX_ENGINE_NAME,
    CodexCLICompletionClient,
    load_codex_engine_config,
    refuse_codex_confirmatory_engine,
    validate_codex_development_profile,
)
from evolution.evaluation import load_task_evaluation_config
from evolution.feedback import load_feedback_profiles
from evolution.gp_operator import PlacementScriptGPOperator, load_gp_operator_config
from evolution.islands import load_islands_config
from evolution.ladder import load_ladder_config
from evolution.llm_operator import OpenWeightLLMOperator, VLLMCompletionClient
from evolution.model import (
    ApiCostLedger,
    host_handshake_sha256,
    load_model_track,
    verify_host_handshake,
    verify_weight_directory,
)
from evolution.openrouter_engine import (
    OpenRouterCompletionClient,
    load_openrouter_engine_config,
)
from evolution.operators import (
    AblationProbeInterleaveOperator,
    SharedNumericInterleaveOperator,
)

ROOT = Path(__file__).parents[1]
CONFIRMATORY_RECEIPT_SCHEMA = "evolved-continuum-confirmatory-postflight-v1"
CANONICAL_INPUTS = {
    "campaign_config": ROOT / "evolution/campaigns.json",
    "task_config": ROOT / "evolution/task/config.json",
    "model_manifest": ROOT / "evolution/open_model.json",
    "initial": ROOT / "evolution/task/initial.py",
    "shinkaevolve_lock": ROOT / "evolution/shinkaevolve.lock.json",
    "campaign_implementation": ROOT / "evolution/campaign.py",
    "evaluation_implementation": ROOT / "evolution/evaluation.py",
    "ladder_implementation": ROOT / "evolution/ladder.py",
    "islands_implementation": ROOT / "evolution/islands.py",
    "model_implementation": ROOT / "evolution/model.py",
    "operators_implementation": ROOT / "evolution/operators.py",
    "llm_operator_implementation": ROOT / "evolution/llm_operator.py",
    "gp_operator_implementation": ROOT / "evolution/gp_operator.py",
    "campaign_cli_implementation": ROOT / "evolution/run_campaign.py",
    "codex_engine_implementation": ROOT / "evolution/codex_engine.py",
    "codex_engine_config": ROOT / "evolution/codex_engine.json",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign_id", required=True)
    parser.add_argument("--arm", choices=("llm", "gp"), required=True)
    parser.add_argument("--archive_dir", type=Path, required=True)
    parser.add_argument("--campaign_config", type=Path, default=ROOT / "evolution/campaigns.json")
    parser.add_argument("--task_config", type=Path, default=ROOT / "evolution/task/config.json")
    parser.add_argument("--model_manifest", type=Path, default=ROOT / "evolution/open_model.json")
    parser.add_argument("--initial", type=Path, default=ROOT / "evolution/task/initial.py")
    parser.add_argument("--proposal_endpoint")
    parser.add_argument(
        "--proposal_engine",
        choices=("open-weight-vllm", CODEX_ENGINE_NAME, "openrouter"),
        default="open-weight-vllm",
    )
    parser.add_argument(
        "--codex_engine_config",
        type=Path,
        default=ROOT / "evolution/codex_engine.json",
    )
    parser.add_argument(
        "--openrouter_engine_config",
        type=Path,
        default=ROOT / "evolution/openrouter_engine.json",
    )
    parser.add_argument("--host_handshake", type=Path)
    parser.add_argument("--weight_directory", type=Path)
    parser.add_argument("--confirmatory_authorization", type=Path)
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow explicitly provisional configs; never use for confirmatory campaigns",
    )
    parser.add_argument(
        "--gp-config",
        type=Path,
        help=(
            "optional GPOperatorConfig JSON; omitted, PlacementScriptGPOperator() "
            "keeps the committed 0.45/0.35/0.20 mix"
        ),
    )
    return parser.parse_args()


def gp_operator_from_args(args: argparse.Namespace) -> PlacementScriptGPOperator:
    """Instantiate the GP proposer. No path keeps the historical constructor call."""
    if args.gp_config is None:
        return PlacementScriptGPOperator()
    config, _label = load_gp_operator_config(args.gp_config)
    return PlacementScriptGPOperator(config)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_confirmatory_receipt(
    path: Path,
    receipt: dict[str, object],
    *,
    engine_descriptor: dict[str, object] | None = None,
) -> None:
    """Durably create or idempotently verify a separate postflight receipt."""
    refuse_codex_confirmatory_engine(
        engine_descriptor, boundary="confirmatory postflight receipt"
    )
    encoded = _canonical(receipt) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise RuntimeError("existing confirmatory postflight receipt differs")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _git_state(path: Path) -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "dirty_entries": git("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
    }


def _exclude_archive_from_git_state(
    state: dict[str, object], archive_dir: str | Path
) -> dict[str, object]:
    """Keep an in-repository development archive from invalidating its own resume.

    EventArchive validates every authoritative archive byte. Repository provenance must
    describe the code/input state, not accumulate newly materialized evidence paths on
    each invocation. Confirmatory archives are already required to live outside both
    repositories, so this only affects the explicitly requested development workflow.
    """
    try:
        relative = Path(archive_dir).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return state
    prefix = relative.rstrip("/") + "/"

    def outside_archive(entry: str) -> bool:
        paths = entry[3:].split(" -> ") if len(entry) >= 4 else [entry]
        return not any(path == relative or path.startswith(prefix) for path in paths)

    return {
        **state,
        "dirty_entries": [
            entry for entry in state["dirty_entries"] if outside_archive(entry)
        ],
    }


def _resolve_frozen_tag(path: Path, tag: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tagged_blob(path: Path, tag: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{tag}:{relative}"],
        cwd=path,
        check=True,
        capture_output=True,
    ).stdout


def _verify_canonical_tagged_inputs(
    supplied: dict[str, Path], *, tag: str, engine_descriptor: dict[str, object] | None = None
) -> dict[str, Path]:
    """Require canonical paths whose bytes equal their frozen-tag blobs."""
    refuse_codex_confirmatory_engine(
        engine_descriptor, boundary="confirmatory frozen-tag verification"
    )
    if set(supplied) != set(CANONICAL_INPUTS):
        raise ValueError("confirmatory input set differs from the registered canonical set")
    verified: dict[str, Path] = {}
    for name, canonical in CANONICAL_INPUTS.items():
        actual = supplied[name].resolve()
        expected = canonical.resolve()
        if actual != expected:
            raise ValueError(f"confirmatory {name} must use the canonical repository path")
        relative = canonical.relative_to(ROOT).as_posix()
        if canonical.read_bytes() != _tagged_blob(ROOT, tag, relative):
            raise ValueError(f"confirmatory {name} differs from frozen tag {tag}")
        verified[name] = canonical
    return verified


def _endpoint_origin(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "proposal endpoint must be an HTTP(S) origin without credentials or a path"
        )
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = "" if parsed.port is None else f":{parsed.port}"
    return f"{parsed.scheme}://{host}{port}"


def _require_external_archive_path(path: str | Path, *, label: str) -> Path:
    """Resolve an evidence path and reject either source checkout as its parent."""
    resolved = Path(path).resolve()
    repositories = (ROOT.resolve(), (ROOT.parent / "ShinkaEvolve").resolve())
    if any(
        resolved == repository or resolved.is_relative_to(repository)
        for repository in repositories
    ):
        raise ValueError(f"{label} must be outside both source repositories")
    return resolved


def main() -> int:
    args = _arguments()
    supplied_inputs = {
        "campaign_config": args.campaign_config,
        "task_config": args.task_config,
        "model_manifest": args.model_manifest,
        "initial": args.initial,
        "codex_engine_config": args.codex_engine_config,
        **{
            name: path
            for name, path in CANONICAL_INPUTS.items()
            if name
            not in {
                "campaign_config",
                "task_config",
                "model_manifest",
                "initial",
                "codex_engine_config",
            }
        },
    }
    plan = load_campaign_plan(args.campaign_config)
    definition = next(
        (campaign for campaign in plan.campaigns if campaign.campaign_id == args.campaign_id),
        None,
    )
    if definition is None:
        raise ValueError(f"unknown campaign_id {args.campaign_id!r}")
    task = load_task_evaluation_config(args.task_config)
    ladder = load_ladder_config(args.task_config)
    islands = load_islands_config(args.task_config)
    profiles = load_feedback_profiles(args.task_config)
    feedback = profiles[plan.feedback_profile]
    track = load_model_track(args.model_manifest)
    codex_config = None
    openrouter_config = None
    engine_descriptor = None
    if args.arm == "gp" and args.proposal_engine != "open-weight-vllm":
        raise ValueError("--proposal_engine applies only to the LLM arm")
    if args.arm == "llm" and args.proposal_engine == CODEX_ENGINE_NAME:
        codex_config = load_codex_engine_config(args.codex_engine_config)
        validate_codex_development_profile(
            codex_config,
            development=args.development,
            campaign_status=plan.status,
            feedback_profile=plan.feedback_profile,
        )
        # The exact runtime version is added after the executable is inspected.  The
        # engine key alone is sufficient for every pre-dispatch confirmatory guard.
        engine_descriptor = {"engine": CODEX_ENGINE_NAME}
    elif args.arm == "llm" and args.proposal_engine == "openrouter":
        openrouter_config = load_openrouter_engine_config(args.openrouter_engine_config)
        engine_descriptor = {
            "engine": "openrouter",
            "model_id": openrouter_config.model_id,
        }
    ready = (
        plan.confirmatory_ready
        and ladder.confirmatory_ready
        and islands.latency_calibration.status == "p1-frozen"
        and track.confirmatory_ready
    )
    if not args.development and not ready:
        raise RuntimeError(
            "confirmatory campaign refused: campaign/ladder/latency/model-host pins "
            "are not all frozen"
        )

    root_state = _exclude_archive_from_git_state(_git_state(ROOT), args.archive_dir)
    sibling = ROOT.parent / "ShinkaEvolve"
    sibling_state = _git_state(sibling)
    lock = json.loads((ROOT / "evolution/shinkaevolve.lock.json").read_text())
    authorization = None
    if not args.development:
        _require_external_archive_path(
            args.archive_dir,
            label="confirmatory archive",
        )
        if args.confirmatory_authorization is None:
            raise ValueError("confirmatory campaign requires its P2 authorization")
        authorization = verify_confirmatory_authorization(
            plan,
            args.confirmatory_authorization,
            engine_descriptor=engine_descriptor,
        )
        _verify_canonical_tagged_inputs(
            supplied_inputs,
            tag=plan.frozen_tag,
            engine_descriptor=engine_descriptor,
        )
        if (
            root_state["head"] != _resolve_frozen_tag(ROOT, plan.frozen_tag)
            or root_state["dirty_entries"]
        ):
            raise RuntimeError("confirmatory campaign requires the frozen clean repository")
        if (
            sibling_state["head"] != lock["fork_commit"]
            or sibling_state["tree"] != lock["fork_tree"]
            or sibling_state["dirty_entries"]
        ):
            raise RuntimeError("confirmatory campaign requires the clean pinned ShinkaEvolve fork")

    endpoint = None
    client = None
    live_handshake = None
    if args.arm == "gp":
        primary_operator = gp_operator_from_args(args)
    elif codex_config is not None:
        if any(
            value is not None
            for value in (
                args.proposal_endpoint,
                args.host_handshake,
                args.weight_directory,
                args.confirmatory_authorization,
            )
        ):
            raise ValueError(
                "Codex CLI engine does not accept endpoint, host, weight, or authorization pins"
            )
        client = CodexCLICompletionClient(codex_config)
        engine_descriptor = client.engine_descriptor
        ledger = ApiCostLedger(0)
        primary_operator = OpenWeightLLMOperator(
            codex_config,
            client,
            ledger,
            proposal_modes=plan.llm_proposal_modes,
            profile=plan.profile,
            ablation_prompt_hint_probability=(
                plan.ablation_prompt_hint_probability
            ),
            proposals_per_call=plan.proposals_per_call,
        )
    elif openrouter_config is not None:
        if any(
            value is not None
            for value in (
                args.proposal_endpoint,
                args.host_handshake,
                args.weight_directory,
            )
        ):
            raise ValueError(
                "OpenRouter engine does not accept endpoint, host, or weight pins"
            )
        ledger = ApiCostLedger(openrouter_config.max_total_microusd)
        client = OpenRouterCompletionClient(openrouter_config)
        primary_operator = OpenWeightLLMOperator(
            openrouter_config,
            client,
            ledger,
            proposal_modes=plan.llm_proposal_modes,
            profile=plan.profile,
            ablation_prompt_hint_probability=(
                plan.ablation_prompt_hint_probability
            ),
            proposals_per_call=plan.proposals_per_call,
        )
    else:
        endpoint = args.proposal_endpoint or os.environ.get("EVOLVED_CONTINUUM_PROPOSAL_ENDPOINT")
        if not endpoint:
            raise ValueError("LLM arm requires --proposal_endpoint")
        endpoint_origin = _endpoint_origin(endpoint)
        ledger = ApiCostLedger(track.max_api_costs_microusd)
        client = VLLMCompletionClient(
            endpoint,
            max_request_cost_microusd=track.max_call_reservation_microusd,
            revision=track.revision,
            weight_set_sha256=track.weight_set_sha256,
        )
        if not args.development and args.weight_directory is None:
            raise ValueError("confirmatory LLM arm requires --weight_directory verification")
        if args.weight_directory is not None:
            verify_weight_directory(track, args.weight_directory)
        live_handshake = client.fetch_handshake()
        verify_host_handshake(track, live_handshake)
        client.bind_handshake(live_handshake)
        if args.host_handshake is not None:
            expected_handshake = json.loads(args.host_handshake.read_text())
            if expected_handshake != live_handshake:
                raise ValueError("local host receipt differs from the live proposal endpoint")
        if authorization is not None and (
            authorization["proposal_endpoint_origin"] != endpoint_origin
            or authorization["proposal_host_handshake_sha256"]
            != host_handshake_sha256(live_handshake)
        ):
            raise ValueError("live proposal endpoint differs from confirmatory authorization")
        primary_operator = OpenWeightLLMOperator(
            track,
            client,
            ledger,
            proposal_modes=plan.llm_proposal_modes,
            profile=plan.profile,
            ablation_prompt_hint_probability=(
                plan.ablation_prompt_hint_probability
            ),
            proposals_per_call=plan.proposals_per_call,
        )
    if plan.ablation_probe_probability > 0.0:
        operator = AblationProbeInterleaveOperator(
            primary_operator,
            probability=plan.ablation_probe_probability,
            shared_numeric_probability=(
                plan.shared_numeric_probability
                if plan.profile.numeric_interleave_enabled
                else 0.0
            ),
        )
    elif plan.profile.numeric_interleave_enabled:
        operator = SharedNumericInterleaveOperator(
            primary_operator,
            probability=plan.shared_numeric_probability,
        )
    else:
        operator = primary_operator

    paths = {name: path.resolve() for name, path in supplied_inputs.items()}
    if args.host_handshake is not None:
        paths["host_handshake"] = args.host_handshake.resolve()
    if args.confirmatory_authorization is not None:
        paths["confirmatory_authorization"] = args.confirmatory_authorization.resolve()
    endpoint_origin = None if endpoint is None else _endpoint_origin(endpoint)
    if codex_config is not None:
        model_provenance = {
            **engine_descriptor,
            "generation": codex_config.generation,
            "max_calls_per_campaign": codex_config.max_calls_per_campaign,
            "max_wall_clock_s_per_campaign": codex_config.max_wall_clock_s_per_campaign,
            "timeout_s": codex_config.timeout_s,
        }
    elif openrouter_config is not None:
        model_provenance = {
            **engine_descriptor,
            "generation": openrouter_config.generation,
            "reasoning_effort": openrouter_config.reasoning_effort,
            "max_output_tokens": openrouter_config.max_output_tokens,
        }
    else:
        model_provenance = {
            "model_id": track.model_id,
            "revision": track.revision,
            "weight_set_sha256": track.weight_set_sha256,
            "host_handshake_sha256": (
                None
                if live_handshake is None
                else host_handshake_sha256(live_handshake)
            ),
            "host_handshake": live_handshake,
            "weights_verified_locally": args.weight_directory is not None,
        }
    provenance = {
        "files": {name: _file_sha256(path) for name, path in paths.items()},
        "model": model_provenance,
        "proposal_endpoint_origin": endpoint_origin,
        "repository": root_state,
        "shinkaevolve_repository": sibling_state,
        "development_override": args.development,
    }
    preflight = {
        "files": provenance["files"],
        "model": provenance["model"],
        "proposal_endpoint_origin": endpoint_origin,
        "repository": root_state,
        "shinkaevolve_repository": sibling_state,
    }
    runner = CampaignRunner(
        definition=definition,
        plan=plan,
        arm=args.arm,
        operator=operator,
        initial_source=args.initial.read_text(),
        task=task,
        ladder=ladder,
        islands=islands,
        feedback_profile=feedback,
        archive_dir=args.archive_dir,
        provenance=provenance,
    )
    digest = runner.run()
    receipt_path = None
    if not args.development:
        post_root_state = _git_state(ROOT)
        post_sibling_state = _git_state(sibling)
        _verify_canonical_tagged_inputs(
            supplied_inputs,
            tag=plan.frozen_tag,
            engine_descriptor=engine_descriptor,
        )
        if args.weight_directory is not None:
            verify_weight_directory(track, args.weight_directory)
        post_handshake = None
        if client is not None and live_handshake is not None:
            post_handshake = client.fetch_handshake()
            verify_host_handshake(track, post_handshake)
        postflight = {
            "files": {name: _file_sha256(path) for name, path in paths.items()},
            "model": {
                **provenance["model"],
                "host_handshake_sha256": (
                    None
                    if post_handshake is None
                    else host_handshake_sha256(post_handshake)
                ),
                "host_handshake": post_handshake,
            },
            "proposal_endpoint_origin": endpoint_origin,
            "repository": post_root_state,
            "shinkaevolve_repository": post_sibling_state,
        }
        if postflight != preflight:
            raise RuntimeError("confirmatory frozen dependencies drifted during execution")
        receipt_path = args.archive_dir.resolve().with_name(
            args.archive_dir.resolve().name + ".confirmatory-postflight.json"
        )
        _write_confirmatory_receipt(
            receipt_path,
            {
                "schema": CONFIRMATORY_RECEIPT_SCHEMA,
                "status": "PASS",
                "campaign_id": definition.campaign_id,
                "arm": args.arm,
                "archive": str(args.archive_dir.resolve()),
                "archive_sha256": digest,
                "preflight": preflight,
                "postflight": postflight,
            },
            engine_descriptor=engine_descriptor,
        )
    print(
        json.dumps(
            {
                "archive_sha256": digest,
                "confirmatory_postflight_receipt": (
                    None if receipt_path is None else str(receipt_path)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
