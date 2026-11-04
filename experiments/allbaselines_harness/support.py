"""Strict input materialization and sealed-artifact helpers for the harness."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from analysis.tracking import derive_seed
from evolution.profiles import (
    ENHANCED_PROFILE,
    MATCHED_PROFILE,
    EvolutionProfile,
    evolution_profile,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG_PATH = HERE / "harness_config.json"
RUNS = HERE / "runs"
# Default evolution seed. `dev-medium-hot` and the older fixtures start every DSL
# arm from this single policy; the dev-continuum-* family needs a different one
# because its dependency-blind policies are infeasible, so the seed is selectable
# per run via `--initial` and recorded in the run manifest.
DEFAULT_INITIAL = ROOT / "evolution/task/initial.py"
INITIAL = DEFAULT_INITIAL
MODEL_MANIFEST = ROOT / "evolution/open_model.json"
DEFAULT_CODEX_CONFIG = ROOT / "experiments/allbaselines_harness/codex_engine_config.json"
DEFAULT_OPENROUTER_CONFIG = ROOT / "evolution/openrouter_engine.json"
BOUNDARY = (
    "Quarantined development comparison only; never acceptance, P1/P2, held-out, "
    "or confirmatory evidence."
)
_SCENARIO_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_DEFAULT_SIZE_GUARD = {
    "max_arm_result_bytes": 10 * 1024 * 1024,
    "max_artifact_bytes": 1024 * 1024 * 1024,
    "max_run_bytes": 4 * 1024 * 1024 * 1024,
    "poll_interval_seconds": 1.0,
}


class RunSizeBudgetExceeded(RuntimeError):
    """A harness artifact crossed a configured fail-closed storage boundary."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def size_guard(config: dict[str, Any]) -> dict[str, Any]:
    """Return the operational storage guard, including defaults for older configs."""
    configured = config.get("artifact_size_guard", {})
    if not isinstance(configured, dict):
        raise ValueError("artifact_size_guard must be a JSON object")
    guard = {**_DEFAULT_SIZE_GUARD, **configured}
    for key in ("max_arm_result_bytes", "max_artifact_bytes", "max_run_bytes"):
        value = guard[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"artifact_size_guard.{key} must be a positive integer")
    poll = guard["poll_interval_seconds"]
    if (
        not isinstance(poll, (int, float))
        or isinstance(poll, bool)
        or not 0.05 <= poll <= 60.0
    ):
        raise ValueError(
            "artifact_size_guard.poll_interval_seconds must be within [0.05, 60]"
        )
    if guard["max_arm_result_bytes"] > guard["max_artifact_bytes"]:
        raise ValueError("arm-result budget cannot exceed the per-artifact budget")
    if guard["max_artifact_bytes"] > guard["max_run_bytes"]:
        raise ValueError("per-artifact budget cannot exceed the run-directory budget")
    compatible = configured.get("resume_compatible_config_sha256", [])
    if not isinstance(compatible, list) or any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in compatible
    ):
        raise ValueError(
            "artifact_size_guard.resume_compatible_config_sha256 must contain SHA-256 strings"
        )
    return guard


def enforce_run_size(run_dir: Path, config: dict[str, Any]) -> int:
    """Fail closed when a file or aggregate run directory exceeds its byte budget."""
    guard = size_guard(config)
    total = 0
    if not run_dir.exists():
        return total
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        artifact_bytes = path.stat().st_size
        if artifact_bytes > guard["max_artifact_bytes"]:
            raise RunSizeBudgetExceeded(
                "artifact size budget exceeded: "
                f"{path} is {artifact_bytes} bytes; configured limit is "
                f"{guard['max_artifact_bytes']} bytes"
            )
        total += artifact_bytes
        if total > guard["max_run_bytes"]:
            raise RunSizeBudgetExceeded(
                "run-directory size budget exceeded: "
                f"{run_dir} is at least {total} bytes; configured limit is "
                f"{guard['max_run_bytes']} bytes (last artifact: {path})"
            )
    return total


def write_exact(path: Path, text: str, *, max_bytes: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text if text.endswith("\n") else text + "\n"
    encoded_bytes = encoded.encode()
    if max_bytes is not None and len(encoded_bytes) > max_bytes:
        raise RunSizeBudgetExceeded(
            "artifact size budget exceeded before write: "
            f"{path} would be {len(encoded_bytes)} bytes; configured limit is "
            f"{max_bytes} bytes"
        )
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"existing harness artifact differs: {path}")
        return
    path.write_bytes(encoded_bytes)


def write_sealed(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    wrapped = {"payload_sha256": canonical_sha256(payload), "payload": payload}
    write_exact(
        path,
        json.dumps(wrapped, sort_keys=True, indent=2),
        max_bytes=max_bytes,
    )
    return wrapped


def sealed_artifact_path(path: Path) -> Path:
    """Resolve a JSON seal, accepting the supervisor's byte-preserving gzip form."""
    compressed = path.with_name(f"{path.name}.gz")
    if path.exists() and compressed.exists():
        raise ValueError(f"ambiguous sealed artifact has JSON and gzip forms: {path}")
    if path.exists():
        return path
    if compressed.exists():
        return compressed
    raise FileNotFoundError(path)


def sealed_artifact_exists(path: Path) -> bool:
    compressed = path.with_name(f"{path.name}.gz")
    if path.exists() and compressed.exists():
        raise ValueError(f"ambiguous sealed artifact has JSON and gzip forms: {path}")
    return path.exists() or compressed.exists()


def read_sealed(path: Path) -> dict[str, Any]:
    resolved = sealed_artifact_path(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rt") as stream:
            wrapped = json.load(stream)
    else:
        wrapped = json.loads(resolved.read_text())
    if set(wrapped) != {"payload", "payload_sha256"}:
        raise ValueError(f"sealed harness artifact has an unknown schema: {resolved}")
    if wrapped["payload_sha256"] != canonical_sha256(wrapped["payload"]):
        raise ValueError(f"sealed harness artifact digest mismatch: {resolved}")
    return wrapped


def verify_sealed_stream(path: Path) -> dict[str, Any]:
    """Verify a pretty-printed seal in constant memory, including ``.json.gz`` files.

    ``write_sealed`` uses sorted, indented JSON. Removing insignificant whitespace from
    its payload therefore reproduces the canonical bytes covered by ``payload_sha256``.
    This avoids materializing multi-GB legacy payloads merely to verify their seal.
    """
    resolved = sealed_artifact_path(path)
    opener = gzip.open if resolved.suffix == ".gz" else Path.open
    digest = hashlib.sha256()
    decompressed_bytes = 0
    with opener(resolved, "rb") as stream:
        first = stream.readline()
        second = stream.readline()
        decompressed_bytes += len(first) + len(second)
        if first.strip() != b"{" or not second.lstrip().startswith(b'"payload":'):
            raise ValueError(f"sealed artifact is not write_sealed JSON: {resolved}")
        colon = second.index(b":")
        start = second.find(b"{", colon + 1)
        if start < 0:
            raise ValueError(f"sealed artifact payload is not a JSON object: {resolved}")
        digest.update(b"{")
        if second[start:].strip() != b"{":
            raise ValueError(f"sealed artifact has unexpected payload formatting: {resolved}")
        for line in stream:
            decompressed_bytes += len(line)
            if line == b"  },\n" or line == b"  },\r\n":
                digest.update(b"}")
                break
            canonical_line = line.lstrip(b" \t").rstrip(b"\r\n")
            if canonical_line.startswith(b'"'):
                escaped = False
                key_end = None
                for index, byte in enumerate(canonical_line[1:], start=1):
                    if escaped:
                        escaped = False
                    elif byte == 0x5C:
                        escaped = True
                    elif byte == 0x22:
                        key_end = index
                        break
                if key_end is not None and canonical_line[key_end + 1 :].startswith(b": "):
                    canonical_line = (
                        canonical_line[: key_end + 2] + canonical_line[key_end + 3 :]
                    )
            digest.update(canonical_line)
        else:
            raise ValueError(f"sealed artifact has an unterminated payload: {resolved}")
        tail = stream.read()
        decompressed_bytes += len(tail)
    match = re.fullmatch(
        rb'\s*"payload_sha256"\s*:\s*"([0-9a-f]{64})"\s*\}\s*', tail
    )
    if match is None:
        raise ValueError(f"sealed artifact lacks payload_sha256: {resolved}")
    claimed = match.group(1).decode()
    actual = digest.hexdigest()
    if claimed != actual:
        raise ValueError(f"sealed harness artifact digest mismatch: {resolved}")
    return {
        "path": str(resolved),
        "payload_sha256": actual,
        "compressed_bytes": resolved.stat().st_size,
        "decompressed_bytes": decompressed_bytes,
    }


def load_harness_config() -> dict[str, Any]:
    raw = json.loads(CONFIG_PATH.read_text())
    if raw.get("schema") != "evolved-continuum-allbaselines-harness-config-v1":
        raise ValueError("unknown all-baselines harness config schema")
    if (
        raw.get("status") != "development-only"
        or raw.get("confirmatory_ready") is not False
        or raw.get("interpretation_boundary") != BOUNDARY
    ):
        raise ValueError("all-baselines harness crosses the development boundary")
    size_guard(raw)
    return raw


def parse_normalized_budget(value: str) -> int:
    """Convert normalized work units to integer microwork without rounding."""
    try:
        normalized = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("budget must be a finite decimal normalized-work value") from exc
    if not normalized.is_finite() or normalized <= 0:
        raise ValueError("budget must be finite and positive")
    microwork = normalized * Decimal(1_000_000)
    if microwork != microwork.to_integral_value():
        raise ValueError("budget must resolve exactly to integer microwork")
    result = int(microwork)
    if result >= 2**63:
        raise ValueError("budget exceeds the signed 64-bit microwork domain")
    return result


def evaluation_seeds(root_seed: int, scenario_id: str, replicates: int) -> tuple[int, ...]:
    return tuple(
        derive_seed(
            root_seed,
            "sim",
            "scenario",
            scenario_id,
            "replicate",
            replicate,
        )
        for replicate in range(replicates)
    )


@dataclass(frozen=True)
class PreparedRun:
    scenario_id: str
    source_task_config: Path
    max_microwork: int
    root_seed: int
    include_llm: bool
    profile_name: str
    ablation_probe_probability: float
    ablation_prompt_hint_probability: float
    proposals_per_call: int
    seed_archive_random_mutants: int | None
    initial_path: Path
    run_dir: Path
    task_config: Path
    campaign_config: Path
    engine: str
    engine_config: Path
    harness_config: dict[str, Any]
    gp_config_path: Path | None = None
    gp_mix_label: str | None = None

    @property
    def codex_config(self) -> Path:
        """Historical alias retained for sealed Codex-run readers and tests."""
        if self.engine != "codex":
            raise ValueError("OpenRouter runs do not have a Codex engine config")
        return self.engine_config

    @property
    def population_parity(self) -> bool:
        return self.seed_archive_random_mutants is not None

    @property
    def initialization_population_size(self) -> int:
        if not self.population_parity:
            return 1
        return len(self.harness_config["seed_archive_reference_policies"]) + int(
            self.seed_archive_random_mutants
        )

    @property
    def evolution_profile(self) -> EvolutionProfile:
        return evolution_profile(self.profile_name)


def required_call_ceiling(config: dict[str, Any], max_microwork: int) -> int:
    """Size the call guard from the worst observed calls/work plus explicit reserve."""
    policy = config.get("llm", {}).get("call_ceiling_policy")
    required = {
        "observed_calls",
        "observed_microwork",
        "safety_factor_numerator",
        "safety_factor_denominator",
        "minimum_calls",
        "maximum_calls",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise ValueError("llm.call_ceiling_policy must contain the exact registered fields")
    if any(
        not isinstance(policy[key], int)
        or isinstance(policy[key], bool)
        or policy[key] < 1
        for key in required
    ):
        raise ValueError("call-ceiling policy fields must be positive integers")
    numerator = (
        max_microwork
        * policy["observed_calls"]
        * policy["safety_factor_numerator"]
    )
    denominator = (
        policy["observed_microwork"] * policy["safety_factor_denominator"]
    )
    observed_requirement = (numerator + denominator - 1) // denominator
    ceiling = max(policy["minimum_calls"], observed_requirement)
    if ceiling > policy["maximum_calls"]:
        raise ValueError("work budget requires more calls than the configured safety maximum")
    return ceiling


def _check_seed(root_seed: int) -> None:
    if (
        not isinstance(root_seed, int)
        or isinstance(root_seed, bool)
        or not 0 <= root_seed < 2**63
    ):
        raise ValueError("root seed must be an explicit integer in [0, 2**63)")


def _check_probability(value: float, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{label} must be a finite number in [0, 1]")
    return float(value)


def _check_quarantined_scenario(scenario_id: str, path: Path) -> None:
    if not scenario_id.startswith("dev-"):
        raise ValueError("harness scenario id must carry the dev- quarantine prefix")
    try:
        relative = path.resolve().relative_to((ROOT / "sim/scenarios").resolve())
    except ValueError as exc:
        raise ValueError("harness scenario must stay under sim/scenarios") from exc
    lowered = [part.lower() for part in relative.parts]
    if not any(
        part == "dev" or part.startswith("dev-") or part.startswith("dev_")
        for part in lowered
    ):
        raise ValueError("harness scenario path is not visibly development-quarantined")
    if {"acceptance", "private", "holdout"} & set(lowered):
        raise ValueError("harness scenario path crosses a protected evidence boundary")


def _check_harness_numbers(config: dict[str, Any], max_microwork: int) -> None:
    rung = config["rung_microwork"]
    if set(rung) != {"L0", "L1", "L2"} or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in rung.values()
    ):
        raise ValueError("harness requires positive integer L0/L1/L2 microwork")
    if max_microwork < sum(rung.values()):
        raise ValueError("matched budget cannot fit one complete L0/L1/L2 candidate")


def _checkpoints(maximum: int) -> list[int]:
    points = {0, maximum}
    points.update((maximum * numerator) // 4 for numerator in (1, 2, 3))
    return sorted(points)


def _write_input_manifest(
    path: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Keep size-guard-only config changes resumable without weakening input pins."""
    if not path.exists():
        write_sealed(path, manifest)
        return
    existing = read_sealed(path)["payload"]
    current_hash = manifest["harness_config_sha256"]
    compatible_hashes = config.get("artifact_size_guard", {}).get(
        "resume_compatible_config_sha256", []
    )
    prior_hash = existing.get("harness_config_sha256")
    existing_without_hash = {
        key: value for key, value in existing.items() if key != "harness_config_sha256"
    }
    manifest_without_hash = {
        key: value for key, value in manifest.items() if key != "harness_config_sha256"
    }
    if (
        existing_without_hash == manifest_without_hash
        and prior_hash in {current_hash, *compatible_hashes}
    ):
        return
    write_sealed(path, manifest)


def prepare_run(
    scenario_id: str,
    task_config: str | Path,
    normalized_budget: str,
    root_seed: int,
    *,
    include_llm: bool,
    profile_name: str = MATCHED_PROFILE,
    ablation_probe_probability: float = 0.0,
    ablation_prompt_hint_probability: float = 0.0,
    proposals_per_call: int = 1,
    seed_archive_random_mutants: int | None = None,
    initial_path: str | Path | None = None,
    runs_root: str | Path | None = None,
    engine: str = "codex",
    engine_config: str | Path | None = None,
    gp_config_path: str | Path | None = None,
) -> PreparedRun:
    config = load_harness_config()
    if engine not in {"codex", "openrouter"}:
        raise ValueError("engine must be codex or openrouter")
    default_engine_config = (
        DEFAULT_CODEX_CONFIG if engine == "codex" else DEFAULT_OPENROUTER_CONFIG
    )
    source_engine_path = (
        default_engine_config
        if engine_config is None
        else (ROOT / Path(engine_config)).resolve()
    )
    try:
        source_engine_path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("engine config must stay inside this repository") from exc
    if not source_engine_path.is_file():
        raise ValueError(f"engine config not found: {source_engine_path}")
    resolved_gp_source: Path | None = None
    gp_mix_label: str | None = None
    gp_operator_config = None
    if gp_config_path is not None:
        from evolution.gp_operator import load_gp_operator_config

        resolved_gp_source = Path(gp_config_path)
        if not resolved_gp_source.is_absolute():
            resolved_gp_source = (ROOT / resolved_gp_source).resolve()
        else:
            resolved_gp_source = resolved_gp_source.resolve()
        try:
            resolved_gp_source.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("GP operator config must stay inside this repository") from exc
        if not resolved_gp_source.is_file():
            raise ValueError(f"GP operator config not found: {resolved_gp_source}")
        gp_operator_config, gp_mix_label = load_gp_operator_config(resolved_gp_source)
    profile = evolution_profile(profile_name)
    resolved_initial = (
        DEFAULT_INITIAL if initial_path is None else (ROOT / Path(initial_path)).resolve()
    )
    if not resolved_initial.is_file():
        raise ValueError(f"evolution seed not found: {resolved_initial}")
    if (
        not isinstance(proposals_per_call, int)
        or isinstance(proposals_per_call, bool)
        or proposals_per_call < 1
    ):
        raise ValueError("proposals_per_call must be a positive integer")
    ablation_probe_probability = _check_probability(
        ablation_probe_probability,
        label="ablation probe share",
    )
    ablation_prompt_hint_probability = _check_probability(
        ablation_prompt_hint_probability,
        label="ablation prompt hint share",
    )
    if profile.shared_numeric_probability + ablation_probe_probability > 1.0:
        raise ValueError("numeric and ablation probe route shares exceed 1")
    if proposals_per_call > 1 and (
        profile.numeric_interleave_enabled or ablation_probe_probability > 0.0
    ):
        raise ValueError(
            "batched proposals require a profile without proposal interleaves"
        )
    if ablation_prompt_hint_probability > 0.0 and not include_llm:
        raise ValueError("ablation prompt hints require --include-llm")
    if profile.name == ENHANCED_PROFILE:
        if seed_archive_random_mutants is None:
            seed_archive_random_mutants = profile.seeded_random_mutants
        elif seed_archive_random_mutants != profile.seeded_random_mutants:
            raise ValueError(
                "enhanced-llm-dev requires exactly two seeded random-AST mutants"
            )
    if not isinstance(scenario_id, str) or _SCENARIO_ID.fullmatch(scenario_id) is None:
        raise ValueError("scenario id contains unsupported characters")
    _check_seed(root_seed)
    max_microwork = parse_normalized_budget(normalized_budget)
    _check_harness_numbers(config, max_microwork)
    if seed_archive_random_mutants is not None:
        if (
            not isinstance(seed_archive_random_mutants, int)
            or isinstance(seed_archive_random_mutants, bool)
            or seed_archive_random_mutants < 1
        ):
            raise ValueError("--seed-archive K requires a positive integer K")
        initialization_size = len(config["seed_archive_reference_policies"]) + (
            seed_archive_random_mutants
        )
        if initialization_size != config["classic_gphh_population_size"]:
            raise ValueError(
                "population parity requires six references plus K mutants to equal "
                "the classic GPHH population size"
            )
        maximum_initialization_work = initialization_size * sum(
            config["rung_microwork"].values()
        )
        if maximum_initialization_work > max_microwork:
            raise ValueError(
                "matched budget cannot fit the complete population-parity initialization"
            )
    source = Path(task_config).resolve()
    try:
        source.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("task config must stay inside this repository") from exc
    raw = json.loads(source.read_text())
    if raw.get("schema") != "evolved-continuum-task-v1":
        raise ValueError("unknown task config schema")
    if raw.get("data_role") != "development-only":
        raise ValueError("harness refuses a non-development task config")
    matches = [item for item in raw.get("scenarios", []) if item.get("id") == scenario_id]
    if len(matches) != 1:
        raise ValueError("named scenario must occur exactly once in the task config")
    selected = matches[0]
    scenario_path = (ROOT / selected["path"]).resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(scenario_path)
    _check_quarantined_scenario(scenario_id, scenario_path)
    seeds = selected.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("named scenario needs explicit development seeds")
    bundles = raw.get("development_bundle_ids")
    if not isinstance(bundles, list) or not bundles or any(
        not isinstance(bundle, str) or not bundle.startswith("dev-") for bundle in bundles
    ):
        raise ValueError("harness task may use development bundles only")

    mode = "all-arms" if include_llm else "non-llm"
    engine_suffix = "" if engine == "codex" else f"--engine-{engine}"
    profile_suffix = "" if profile_name == MATCHED_PROFILE else f"--{profile_name}"
    ablation_suffix = ""
    if ablation_probe_probability > 0.0:
        token = format(ablation_probe_probability, ".12g").replace(".", "p")
        ablation_suffix += f"--ablation-probe-{token}"
    if ablation_prompt_hint_probability > 0.0:
        token = format(ablation_prompt_hint_probability, ".12g").replace(".", "p")
        ablation_suffix += f"--ablation-prompt-{token}"
    parity_suffix = (
        ""
        if seed_archive_random_mutants is None
        else f"--seed-archive-{seed_archive_random_mutants}"
    )
    batch_suffix = "" if proposals_per_call == 1 else f"--batch-{proposals_per_call}"
    gp_mix_suffix = "" if gp_mix_label is None else f"--gp-mix-{gp_mix_label}"
    selected_runs_root = RUNS if runs_root is None else Path(runs_root).resolve()
    if runs_root is not None:
        try:
            selected_runs_root.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("runs root must stay inside this repository") from exc
    run_dir = selected_runs_root / (
        f"{scenario_id}--seed-{root_seed}--work-{max_microwork}--{mode}"
        f"{profile_suffix}{engine_suffix}{parity_suffix}{batch_suffix}"
        f"{ablation_suffix}{gp_mix_suffix}"
    )
    enforce_run_size(run_dir, config)
    inputs = run_dir / "inputs"
    runtime_task = json.loads(json.dumps(raw))
    if profile.name == ENHANCED_PROFILE:
        enhanced_feedback = runtime_task["feedback_profiles"].setdefault(
            ENHANCED_PROFILE, {"enabled": True}
        )
        if enhanced_feedback != {"enabled": True}:
            raise ValueError("enhanced feedback profile must be enabled exactly")
    runtime_task["scenarios"] = [selected]
    runtime_task["task_variant"] = f"allbaselines:{scenario_id}"
    runtime_task["task_variant_note"] = BOUNDARY
    runtime_task["ladder"]["l1"]["scenario_ids"] = [scenario_id]
    runtime_task["ladder"]["budget_registration"] = {
        "status": "development-provisional",
        "source": "allbaselines harness: quarantined development comparison",
    }
    task_path = inputs / "task_config.json"
    write_exact(task_path, json.dumps(runtime_task, sort_keys=True, indent=2))

    roots = [root_seed]
    for index in range(1, 10):
        roots.append(derive_seed(root_seed, "allbaselines", "campaign-slot", index))
    if len(set(roots)) != len(roots):
        raise ValueError("derived campaign roots collided")
    rung = config["rung_microwork"]
    campaign = {
        "schema": "evolved-continuum-campaigns-v3",
        "status": "development-provisional",
        "frozen_tag": None,
        "confirmatory_authorization_sha256": None,
        "candidate_budget": max_microwork // min(rung.values()) + 2,
        "population_per_island": 8,
        "max_operator_attempts": 3,
        "shared_numeric_probability": profile.shared_numeric_probability,
        "llm_proposal_modes": ["diff", "rewrite", "crossover"],
        "evaluator_work": {
            "registration_status": "development-provisional",
            "registration_source": (
                "allbaselines harness: user-selected matched normalized-work budget; "
                "development-provisional rung weights"
            ),
            "max_microwork": max_microwork,
            "rung_microwork": rung,
            "checkpoints_microwork": _checkpoints(max_microwork),
        },
        "feedback_profile": profile.feedback_profile,
        "evolution_profile": profile_name,
        "arms": ["llm", "gp"],
        "campaigns": [
            {
                "campaign_id": f"h{index:02d}",
                "root_seed": seed,
                "evaluation_replicates": config["evaluation_replicates"],
            }
            for index, seed in enumerate(roots)
        ],
    }
    if ablation_probe_probability > 0.0:
        campaign["ablation_probe_probability"] = ablation_probe_probability
    if ablation_prompt_hint_probability > 0.0:
        campaign["ablation_prompt_hint_probability"] = (
            ablation_prompt_hint_probability
        )
    if proposals_per_call > 1:
        campaign["proposals_per_call"] = proposals_per_call
    campaign_path = inputs / "campaign_config.json"
    write_exact(campaign_path, json.dumps(campaign, sort_keys=True, indent=2))
    runtime_engine = json.loads(source_engine_path.read_text())
    if engine == "codex":
        if profile.name == ENHANCED_PROFILE:
            enhanced_policy = runtime_engine["profile_policy"].setdefault(
                ENHANCED_PROFILE, "development-only"
            )
            if enhanced_policy != "development-only":
                raise ValueError("enhanced Codex profile policy must be development-only")
        runtime_engine["max_calls_per_campaign"] = required_call_ceiling(
            config, max_microwork
        )
    engine_path = inputs / f"{engine}_engine_config.json"
    write_exact(engine_path, json.dumps(runtime_engine, sort_keys=True, indent=2))
    manifest = {
        "schema": "evolved-continuum-allbaselines-run-inputs-v1",
        "status": "development-only",
        "confirmatory_ready": False,
        "interpretation_boundary": BOUNDARY,
        "scenario_id": scenario_id,
        "scenario_path": str(scenario_path.relative_to(ROOT)),
        "scenario_sha256": file_sha256(scenario_path),
        "source_task_config": str(source.relative_to(ROOT)),
        "source_task_config_sha256": file_sha256(source),
        "runtime_task_config_sha256": file_sha256(task_path),
        "campaign_config_sha256": file_sha256(campaign_path),
        "harness_config_sha256": file_sha256(CONFIG_PATH),
        "root_seed": root_seed,
        "evaluation_seeds": list(
            evaluation_seeds(
                root_seed,
                scenario_id,
                config["evaluation_replicates"],
            )
        ),
        "matched_max_microwork": max_microwork,
        "matched_normalized_work": max_microwork / 1_000_000,
        "include_llm": include_llm,
        "evolution_profile": profile_name,
        "initialization_mode": (
            "single-seed" if seed_archive_random_mutants is None else "population-parity"
        ),
        "seed_archive_random_mutants": seed_archive_random_mutants,
        "initialization_population_size": (
            1
            if seed_archive_random_mutants is None
            else len(config["seed_archive_reference_policies"])
            + seed_archive_random_mutants
        ),
    }
    if engine == "codex":
        manifest.update(
            {
                "source_codex_engine_config_sha256": file_sha256(source_engine_path),
                "codex_engine_config_sha256": file_sha256(engine_path),
            }
        )
    else:
        manifest.update(
            {
                "engine": engine,
                "source_engine_config_sha256": file_sha256(source_engine_path),
                "engine_config_sha256": file_sha256(engine_path),
            }
        )
    if ablation_probe_probability > 0.0 or ablation_prompt_hint_probability > 0.0:
        manifest["ablation_controls"] = {
            "ablation_probe_probability": ablation_probe_probability,
            "ablation_prompt_hint_probability": ablation_prompt_hint_probability,
        }
    if proposals_per_call > 1:
        manifest["proposals_per_call"] = proposals_per_call
    runtime_gp_path: Path | None = None
    if gp_operator_config is not None:
        assert resolved_gp_source is not None
        assert gp_mix_label is not None
        runtime_gp_path = inputs / "gp_operator_config.json"
        write_exact(
            runtime_gp_path,
            json.dumps(
                {
                    "label": gp_mix_label,
                    "crossover_probability": gp_operator_config.crossover_probability,
                    "subtree_mutation_probability": (
                        gp_operator_config.subtree_mutation_probability
                    ),
                    "numeric_mutation_probability": (
                        gp_operator_config.numeric_mutation_probability
                    ),
                    "max_attempts": gp_operator_config.max_attempts,
                },
                sort_keys=True,
                indent=2,
            ),
        )
        manifest["gp_mix_label"] = gp_mix_label
        manifest["source_gp_operator_config"] = str(
            resolved_gp_source.relative_to(ROOT)
        )
        manifest["source_gp_operator_config_sha256"] = file_sha256(resolved_gp_source)
        manifest["gp_operator_config_sha256"] = file_sha256(runtime_gp_path)
    _write_input_manifest(inputs / "input_manifest.json", manifest, config)
    enforce_run_size(run_dir, config)
    return PreparedRun(
        scenario_id=scenario_id,
        source_task_config=source,
        max_microwork=max_microwork,
        root_seed=root_seed,
        include_llm=include_llm,
        profile_name=profile_name,
        ablation_probe_probability=ablation_probe_probability,
        ablation_prompt_hint_probability=ablation_prompt_hint_probability,
        proposals_per_call=proposals_per_call,
        seed_archive_random_mutants=seed_archive_random_mutants,
        initial_path=resolved_initial,
        run_dir=run_dir,
        task_config=task_path,
        campaign_config=campaign_path,
        engine=engine,
        engine_config=engine_path,
        harness_config=config,
        gp_config_path=runtime_gp_path,
        gp_mix_label=gp_mix_label,
    )
