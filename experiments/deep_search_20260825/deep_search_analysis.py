"""Cross-root comparison on the endpoints that survive their own noise.

Endpoint choice, and why it is not best-of-run
---------------------------------------------
``experiments/power_analysis_20260823`` and ``experiments/estimand_power_20260824``
measured what the 2026-08 campaigns could actually support. Best-of-run search gain has
a paired root SD near 0.049-0.058 against effects near 0.02, which prices the
LLM-vs-GP contrast at 63 roots for 90% power and the LLM-vs-random contrast at 38. Of
six claims made from single roots, exactly one survived re-analysis — and it was the
distributional one, because a maximum statistic inherits the whole tail.

So this module reports three endpoints and is explicit about which is primary:

``archive_median`` (**primary quality**)
    Median normalized energy over the arm's feasible archive at a fixed work budget.
    Paired root SD 0.0104 in the pilot, which prices it at ~5 roots at the point
    estimate and 3-30 across the conditional SD interval. It answers "typical candidate
    quality at a fixed budget", not "best discoverable policy", and that limit is
    stated rather than blurred.

``improved`` (**primary reliability**)
    Did the arm improve on its own initialization at all? This was the most robust
    signal in the whole pilot — the LLM improved on 6 of 6 runs while same-DSL GP
    managed 4 of 7 and random-AST 5 of 6 — and it is the property an operator actually
    depends on. A method that is occasionally brilliant and occasionally returns nothing
    is not a method you deploy.

``best_absolute`` (**descriptive only, never a headline**)
    Reported because readers will ask, and labelled so nobody reads a rank off it.

Development-only. Nothing here is confirmatory: Amendment A1 forbids a confirmatory
number from the subscription engine regardless of what these numbers say.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

def profile_of(run_name: str) -> str:
    """The evolution profile a run directory was produced under.

    Run directories for every profile live side by side under `runs/`, so a glob that
    does not filter would silently pool `deep-search` and `deep-search-norepair` roots
    into one comparison -- averaging the treatment with its own control. The directory
    name carries the profile between `--all-arms--` and `--seed-archive`.
    """
    marker = "--all-arms--"
    if marker not in run_name:
        return "unknown"
    tail = run_name.split(marker, 1)[1]
    profile = tail.split("--seed-archive", 1)[0]
    return profile.split("--engine-", 1)[0]


def engine_of(run_name: str) -> str:
    """Selected proposal engine; absent means the historical Codex default."""
    marker = "--engine-"
    if marker not in run_name:
        return "codex"
    return run_name.split(marker, 1)[1].split("--", 1)[0]


def runs_for_profile(
    runs_root: Path,
    profile: str,
    engine: str = "codex",
    excluded_seeds: set[int] | frozenset[int] = frozenset(),
) -> list[Path]:
    return sorted(
        path
        for path in runs_root.glob("*")
        if (path / "report.json").is_file()
        and profile_of(path.name) == profile
        and engine_of(path.name) == engine
        and root_seed_of(path) not in excluded_seeds
    )


def root_seed_of(run_dir: Path) -> int:
    """Read the root seed from a completed run's sealed or unsealed report."""
    body = json.loads((run_dir / "report.json").read_text())
    body = body.get("payload", body)
    return int(body.get("run", {}).get("root_seed", 0))


ADAPTIVE_ARMS = ("llm", "same_dsl_gp", "random_ast", "classic_gphh")
PRIMARY_ARM = "llm"


@dataclass(frozen=True)
class ProposalFunnel:
    """Per-arm proposal accounting, which plan §2.3 requires and nothing reported.

    The LLM arm's grammar-aware repair is asymmetric by construction: only
    `evolution.llm_operator` emits the `validation_failure` metadata the runner needs to
    build a repair prompt, and a deterministic operator has no error channel to feed. It
    is *not* extra budget -- `max_operator_attempts` is a plan-level constant (3),
    identical for every arm, and a repair replaces a blind retry inside that same loop
    rather than adding one. What differs is how an operator spends its own three
    attempts. That is a real difference between proposal operators and it belongs in the
    comparison; what it must not be is invisible, so it is counted here.
    """

    attempts: int
    invalid_attempts: int
    repair_attempts: int
    valid_candidates: int

    @property
    def attempts_per_valid(self) -> float | None:
        if self.valid_candidates == 0:
            return None
        return self.attempts / self.valid_candidates

    @property
    def invalid_fraction(self) -> float | None:
        if self.attempts == 0:
            return None
        return self.invalid_attempts / self.attempts

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["attempts_per_valid"] = self.attempts_per_valid
        payload["invalid_fraction"] = self.invalid_fraction
        return payload


def read_funnel(run_dir: Path, arm: str) -> ProposalFunnel | None:
    """Count proposal attempts, invalid attempts and repairs from an arm's archive."""
    archive = {"llm": "llm", "same_dsl_gp": "same_dsl_gp"}.get(arm)
    if archive is None:
        return None
    # Archives from completed runs are routinely gzipped to reclaim disk; open either
    # form rather than making every reader decompress by hand first.
    path = run_dir / "archives" / archive / "events.jsonl"
    opener: Any = path.open
    if not path.is_file():
        packed = path.with_suffix(".jsonl.gz")
        if not packed.is_file():
            return None
        opener = functools.partial(gzip.open, packed, "rt")
    attempts = invalid = repairs = valid = 0
    with opener(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "candidate_completed":
                continue
            proposal = event.get("payload", {}).get("proposal", {})
            records = proposal.get("attempts") or []
            attempts += len(records)
            invalid += sum(1 for item in records if item.get("valid") is False)
            repairs += sum(1 for item in records if item.get("repair_of_attempt") is not None)
            if proposal.get("source"):
                valid += 1
    return ProposalFunnel(attempts, invalid, repairs, valid)


@dataclass(frozen=True)
class ArmRoot:
    run: str
    root_seed: int
    arm: str
    candidates: int
    feasible: int
    archive_median: float | None
    archive_mean: float | None
    best_absolute: float | None
    best_at_initialization: float | None
    improved: bool
    funnel: ProposalFunnel | None = None
    ladder: LadderFidelity | None = None

    @property
    def search_delta(self) -> float | None:
        if self.best_absolute is None or self.best_at_initialization is None:
            return None
        return self.best_at_initialization - self.best_absolute

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["search_delta"] = self.search_delta
        payload["funnel"] = None if self.funnel is None else self.funnel.to_dict()
        payload["ladder"] = None if self.ladder is None else self.ladder.to_dict()
        return payload


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def read_run(run_dir: Path) -> list[ArmRoot]:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return []
    body = json.loads(report_path.read_text())
    body = body.get("payload", body)
    run_meta = body.get("run", {})
    root_seed = int(run_meta.get("root_seed", 0))
    out: list[ArmRoot] = []
    for arm in ADAPTIVE_ARMS:
        arm_body = body.get("arms", {}).get(arm)
        if not isinstance(arm_body, dict):
            continue
        energies = [
            value
            for value in (
                _finite(row.get("candidate_normalized_energy"))
                for row in arm_body.get("trajectory", [])
            )
            if value is not None
        ]
        best = arm_body.get("best_absolute")
        start = arm_body.get("best_at_initialization")
        out.append(
            ArmRoot(
                run=run_dir.name,
                root_seed=root_seed,
                arm=arm,
                candidates=int(arm_body.get("candidate_count", len(energies))),
                feasible=len(energies),
                archive_median=statistics.median(energies) if energies else None,
                archive_mean=statistics.fmean(energies) if energies else None,
                best_absolute=_finite(best) if not isinstance(best, dict) else _finite(
                    best.get("normalized_energy")
                ),
                best_at_initialization=(
                    _finite(start)
                    if not isinstance(start, dict)
                    else _finite(start.get("normalized_energy"))
                ),
                improved=bool(arm_body.get("search_improved", False)),
                funnel=read_funnel(run_dir, arm),
                ladder=read_ladder_fidelity(run_dir, arm),
            )
        )
    return out


# Inference seed. The repository forbids entropy defaults (docs/SEEDS.md), and a
# bootstrap whose resamples move between runs is not a reported number.
INFERENCE_SEED = 20260826


# Hand-written references, used to locate the *feasible* frontier a search must beat.
HANDWRITTEN_REFERENCES = (
    "energy_greedy",
    "first_fit",
    "latency_greedy",
    "least_allocated",
    "most_allocated",
    "network_overhead",
)


def opportunity_gaps(runs: list[Path]) -> dict[str, Any]:
    """Re-derive the minimum relevant effect on *this* scenario.

    `ESTIMAND_POWER.md` defines the MRE as 0.20 x the mean opportunity gap, where the
    gap is the best feasible hand-written policy minus the best observed adaptive
    policy, per root. Its value of 0.024148 was computed on `dev-medium-hot`, whose
    feasible energies span roughly 1.33-1.40. This family spans roughly 0.64-1.27, so
    carrying that MRE across would price a *relevant* effect at a fraction of what the
    scenario can actually express -- and would flatter any result measured against it.
    The paired SD moved by the same token (0.0104 on the old fixture, 0.185 here), so
    both halves of the power calculation need re-deriving together.
    """
    gaps: list[float] = []
    per_root: list[dict[str, Any]] = []
    for run in runs:
        body = json.loads((run / "report.json").read_text())
        body = body.get("payload", body)
        arms = body.get("arms", {})
        references = [
            _finite(arms.get(name, {}).get("best_absolute"))
            for name in HANDWRITTEN_REFERENCES
            if isinstance(arms.get(name), dict)
        ]
        adaptive = [
            _finite(arms.get(name, {}).get("best_absolute"))
            for name in ADAPTIVE_ARMS
            if isinstance(arms.get(name), dict)
        ]
        best_reference = min((v for v in references if v is not None), default=None)
        best_adaptive = min((v for v in adaptive if v is not None), default=None)
        if best_reference is None or best_adaptive is None:
            continue
        gap = best_reference - best_adaptive
        gaps.append(gap)
        per_root.append(
            {
                "run": run.name,
                "best_feasible_reference": best_reference,
                "best_observed_adaptive": best_adaptive,
                "opportunity_gap": gap,
            }
        )
    if not gaps:
        return {"roots": 0, "mean_opportunity_gap": None, "minimum_relevant_effect": None}
    mean_gap = statistics.fmean(gaps)
    return {
        "roots": len(gaps),
        "per_root": per_root,
        "mean_opportunity_gap": mean_gap,
        "minimum_relevant_effect": 0.20 * mean_gap,
        "mre_rule": "0.20 x mean opportunity gap (ESTIMAND_POWER.md), re-derived here",
        "superseded_mre": 0.024148,
        "superseded_source": "ESTIMAND_POWER.md, computed on dev-medium-hot",
    }


def roots_for_power(effect: float, sd: float, power: float = 0.90) -> int | None:
    """Paired-t sample size for a given effect and SD; normal approximation."""
    if not effect or not sd or effect <= 0 or sd <= 0:
        return None
    # z(0.975) + z(0.90) = 1.959964 + 1.281552
    return max(2, math.ceil(((1.959964 + 1.281552) * sd / effect) ** 2))


def paired_inference(deltas: list[float], *, resamples: int = 10000) -> dict[str, Any]:
    """Paired bootstrap CI and an exact sign-flip randomization p-value.

    Plan §6.3 prescribes paired bootstrap/randomization inference, and nothing was
    computing it: the analysis reported means and SDs only. That is precisely how the
    2026-08 effort produced six claims from descriptive means of which one survived
    re-analysis, so the test belongs next to the mean rather than in a later pass.

    The randomization test is exact for small n -- it enumerates all 2^n sign flips when
    that is cheaper than sampling, which at eight roots it is (256 assignments). Under
    the null the pairing carries no signal, so flipping the sign of any subset of paired
    differences is equally likely.
    """
    n = len(deltas)
    if n < 2:
        return {"n": n, "mean": deltas[0] if deltas else None, "ci95": None, "p_value": None}
    mean = statistics.fmean(deltas)

    rng = random.Random(INFERENCE_SEED)
    boots = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.fmean(sample))
    boots.sort()
    low = boots[int(0.025 * resamples)]
    high = boots[min(resamples - 1, int(0.975 * resamples))]

    observed = abs(mean)
    if n <= 20:  # exact enumeration
        extreme = 0
        for mask in range(1 << n):
            flipped = [
                -value if (mask >> index) & 1 else value for index, value in enumerate(deltas)
            ]
            if abs(statistics.fmean(flipped)) >= observed - 1e-12:
                extreme += 1
        p_value = extreme / (1 << n)
        exact = True
    else:
        extreme = 0
        for _ in range(resamples):
            flipped = [value if rng.random() < 0.5 else -value for value in deltas]
            if abs(statistics.fmean(flipped)) >= observed - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (resamples + 1)
        exact = False
    return {
        "n": n,
        "mean": mean,
        "ci95": [low, high],
        "p_value": p_value,
        "randomization_exact": exact,
        "seed": INFERENCE_SEED,
    }


def mcnemar_exact(rows: list[ArmRoot], primary: str, comparator: str) -> dict[str, Any]:
    """Exact paired sign test on *whether each arm improved at all*, blocked by root.

    The reliability endpoint is binary, so the paired quantity is the discordant pairs:
    roots where one arm improved and the other did not. Under the null those split 50/50.
    """
    by_root: dict[int, dict[str, ArmRoot]] = {}
    for row in rows:
        by_root.setdefault(row.root_seed, {})[row.arm] = row
    only_primary = only_comparator = concordant = 0
    for arms in by_root.values():
        a, b = arms.get(primary), arms.get(comparator)
        if a is None or b is None:
            continue
        if a.improved and not b.improved:
            only_primary += 1
        elif b.improved and not a.improved:
            only_comparator += 1
        else:
            concordant += 1
    discordant = only_primary + only_comparator
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(only_primary, only_comparator)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2 * tail / (2 ** discordant))
    return {
        "improved_only_primary": only_primary,
        "improved_only_comparator": only_comparator,
        "concordant": concordant,
        "p_value": p_value,
    }


def paired_contrasts(rows: list[ArmRoot], endpoint: str) -> dict[str, dict[str, Any]]:
    """Paired ``PRIMARY_ARM`` minus comparator on one endpoint, blocked by root.

    Pairing is by root seed with common random numbers, which is the whole reason the
    pilot's median SD was an order of magnitude below its best-of-run SD. Roots where
    either side is missing the endpoint are dropped and counted, never imputed.
    """
    by_root: dict[int, dict[str, ArmRoot]] = {}
    for row in rows:
        by_root.setdefault(row.root_seed, {})[row.arm] = row

    out: dict[str, dict[str, Any]] = {}
    for comparator in ADAPTIVE_ARMS:
        if comparator == PRIMARY_ARM:
            continue
        deltas: list[float] = []
        dropped = 0
        for _root, arms in sorted(by_root.items()):
            primary, other = arms.get(PRIMARY_ARM), arms.get(comparator)
            if primary is None or other is None:
                dropped += 1
                continue
            a, b = getattr(primary, endpoint), getattr(other, endpoint)
            if a is None or b is None:
                dropped += 1
                continue
            deltas.append(float(a) - float(b))
        out[comparator] = {
            "inference": paired_inference(deltas),
            "paired_roots": len(deltas),
            "dropped_roots": dropped,
            "mean": statistics.fmean(deltas) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "sd": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "wins_for_primary": sum(1 for value in deltas if value < 0),
            "deltas": deltas,
        }
    return out


@dataclass(frozen=True)
class LadderFidelity:
    """False-elimination evidence for the L1 rejection threshold.

    `dev-continuum-tight` sets `max_summed_primitive_shortfall = 1.0` at L1 so the
    catastrophically infeasible shapes are dropped for 0.4 work instead of buying a full
    L2. That threshold was chosen by reasoning, not measurement, and a threshold that is
    too tight throttles every arm's search -- possibly unequally, since the arms produce
    different candidate distributions. The ladder's epsilon-audit already re-evaluates a
    registered fraction of rejects one rung up, which is exactly the measurement; it had
    simply never fired, because the previous fixtures' threshold of 3.0 rejected nothing.

    A `false elimination` is an L1-rejected candidate the audit found **feasible** at L2.
    """

    l1_rejects: int
    audited: int
    audited_feasible_at_l2: int
    audited_energies: tuple[float, ...]

    @property
    def false_elimination_rate(self) -> float | None:
        if self.audited == 0:
            return None
        return self.audited_feasible_at_l2 / self.audited

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["audited_energies"] = list(self.audited_energies)
        payload["false_elimination_rate"] = self.false_elimination_rate
        return payload


def read_ladder_fidelity(run_dir: Path, arm: str) -> LadderFidelity | None:
    """Count L1 rejects and what the epsilon-audit found when it re-ran them at L2."""
    archive = {"llm": "llm", "same_dsl_gp": "same_dsl_gp"}.get(arm)
    if archive is None:
        return None
    path = run_dir / "archives" / archive / "events.jsonl"
    opener: Any = path.open
    if not path.is_file():
        packed = path.with_suffix(".jsonl.gz")
        if not packed.is_file():
            return None
        opener = functools.partial(gzip.open, packed, "rt")
    rejects = audited = feasible = 0
    energies: list[float] = []
    with opener(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "candidate_completed":
                continue
            evaluation = event.get("payload", {}).get("evaluation")
            if not isinstance(evaluation, dict):
                continue
            for record in evaluation.get("trace", []):
                if record.get("rung") == "L1" and record.get("decision") == "reject":
                    rejects += 1
                if record.get("rung") != "epsilon-audit" or record.get("rejected_at") != "L1":
                    continue
                if not record.get("selected"):
                    continue
                audited += 1
                result = record.get("result") or {}
                fitness = (result.get("evaluation") or {}).get("fitness") or {}
                if fitness.get("feasible"):
                    feasible += 1
                    energy = _finite(fitness.get("normalized_energy"))
                    if energy is not None:
                        energies.append(energy)
    return LadderFidelity(rejects, audited, feasible, tuple(energies))


def ladder_summary(rows: list[ArmRoot]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in ADAPTIVE_ARMS:
        items = [row.ladder for row in rows if row.arm == arm and row.ladder is not None]
        if not items:
            continue
        rejects = sum(item.l1_rejects for item in items)
        audited = sum(item.audited for item in items)
        feasible = sum(item.audited_feasible_at_l2 for item in items)
        out[arm] = {
            "l1_rejects": rejects,
            "audited": audited,
            "audited_feasible_at_l2": feasible,
            "false_elimination_rate": (feasible / audited) if audited else None,
        }
    return out


def funnel_summary(rows: list[ArmRoot]) -> dict[str, dict[str, Any]]:
    """Aggregate proposal accounting per arm across roots."""
    out: dict[str, dict[str, Any]] = {}
    for arm in ADAPTIVE_ARMS:
        funnels = [row.funnel for row in rows if row.arm == arm and row.funnel is not None]
        if not funnels:
            continue
        attempts = sum(item.attempts for item in funnels)
        valid = sum(item.valid_candidates for item in funnels)
        out[arm] = {
            "roots_with_archive": len(funnels),
            "attempts": attempts,
            "invalid_attempts": sum(item.invalid_attempts for item in funnels),
            "repair_attempts": sum(item.repair_attempts for item in funnels),
            "valid_candidates": valid,
            "attempts_per_valid": (attempts / valid) if valid else None,
            "shared_attempt_budget_per_candidate": 3,
        }
    return out


def reliability(rows: list[ArmRoot]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in ADAPTIVE_ARMS:
        arm_rows = [row for row in rows if row.arm == arm]
        if not arm_rows:
            continue
        improved = sum(1 for row in arm_rows if row.improved)
        deltas = [row.search_delta for row in arm_rows if row.search_delta is not None]
        out[arm] = {
            "roots": len(arm_rows),
            "improved": improved,
            "improvement_rate": improved / len(arm_rows),
            "mean_search_delta": statistics.fmean(deltas) if deltas else None,
            "min_search_delta": min(deltas) if deltas else None,
            "sd_search_delta": statistics.stdev(deltas) if len(deltas) > 1 else None,
        }
    return out


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seal_payload(payload: Any) -> dict[str, Any]:
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    }
