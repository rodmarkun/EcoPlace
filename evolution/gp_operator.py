"""Grammar-guided GP proposal operator over the PlacementScript AST (TODO 5.2).

This is the controlled-ablation operator, not the external classic-GPHH baseline.
It consumes and emits the identical PlacementScript representation used by the LLM
arm and implements three ordinary tree-search moves through the native Phase-1 AST
API: typed crossover, same-parent subtree mutation, and numeric-literal mutation.
All randomness arrives through the runner-owned ``random.Random`` argument.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import placementscript as ps
from shinka.core import ProposalOffspring
from shinka.database import Program

_GP_MIX_LABEL = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_GP_CONFIG_KEYS = frozenset(
    {
        "crossover_probability",
        "subtree_mutation_probability",
        "numeric_mutation_probability",
        "max_attempts",
        "label",
    }
)


@dataclass(frozen=True)
class GPOperatorConfig:
    crossover_probability: float = 0.45
    subtree_mutation_probability: float = 0.35
    numeric_mutation_probability: float = 0.20
    max_attempts: int = 24

    def __post_init__(self) -> None:
        probabilities = (
            self.crossover_probability,
            self.subtree_mutation_probability,
            self.numeric_mutation_probability,
        )
        if any(not math.isfinite(p) or p < 0.0 for p in probabilities):
            raise ValueError("GP operation probabilities must be finite and non-negative")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("GP operation probabilities must sum to 1")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive int")


def _require_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    return float(value)


def load_gp_operator_config(path: str | Path) -> tuple[GPOperatorConfig, str]:
    """Load an optional on-disk GP mix. Unknown keys fail closed."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GP operator config must be a JSON object")
    unknown = sorted(set(raw) - _GP_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown GP operator config keys: {unknown}")
    required = (
        "crossover_probability",
        "subtree_mutation_probability",
        "numeric_mutation_probability",
        "max_attempts",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"GP operator config missing keys: {missing}")
    config = GPOperatorConfig(
        crossover_probability=_require_probability(
            raw["crossover_probability"], "crossover_probability"
        ),
        subtree_mutation_probability=_require_probability(
            raw["subtree_mutation_probability"], "subtree_mutation_probability"
        ),
        numeric_mutation_probability=_require_probability(
            raw["numeric_mutation_probability"], "numeric_mutation_probability"
        ),
        max_attempts=raw["max_attempts"],
    )
    label = raw.get("label", source.stem)
    if not isinstance(label, str) or _GP_MIX_LABEL.fullmatch(label) is None:
        raise ValueError("GP operator config label must be lowercase hyphenated tokens")
    return config, label


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _mutable_sites(source: str) -> list[dict[str, Any]]:
    # Module constants carry artifact identity and stay immutable. Every function AST
    # node remains in the hypothesis space, including full score/proposer functions.
    return [dict(site) for site in ps.enumerate_subtrees(source) if site["func"] is not None]


def _compatible(target: dict[str, Any], donor: dict[str, Any]) -> bool:
    if target["is_expr"] != donor["is_expr"]:
        return False
    if target["is_expr"]:
        return target["type"] == donor["type"]
    if target["kind"] == "func" or donor["kind"] == "func":
        return target["kind"] == donor["kind"] and target["func"] == donor["func"]
    return True


def _replacement_float(value: float, rng: random.Random) -> str:
    if value == 0.0:
        changed = rng.choice((-1.0, 1.0)) * rng.choice((1e-3, 1e-2, 1e-1))
    else:
        changed = value * rng.choice((0.5, 0.75, 0.9, 1.1, 1.25, 2.0))
    text = f"{changed:.12g}"
    return text if "." in text or "e" in text.lower() else f"{text}.0"


def _replacement_int(value: int, rng: random.Random) -> str:
    if value == 0:
        return str(rng.choice((1, 2)))
    step = max(1, round(abs(value) * rng.choice((0.1, 0.25, 0.5))))
    changed = value + rng.choice((-step, step))
    # The sign of an ordinary negative literal belongs to a unary AST node, so the
    # literal hook accepts unsigned replacements there. Existing positive knobs stay
    # non-negative to avoid turning counts/limits into malformed policies gratuitously.
    return str(max(0, changed)) if value >= 0 else str(abs(changed))


class PlacementScriptGPOperator:
    """Seeded grammar-guided proposer conforming to ShinkaEvolve's common contract."""

    name = "same_dsl_gp"

    def __init__(self, config: GPOperatorConfig | None = None) -> None:
        self.config = config or GPOperatorConfig()

    def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("GP proposer requires the runner-owned random.Random")
        try:
            ps.Policy(parent.code)
        except ps.PlacementScriptError as exc:
            return ProposalOffspring(
                source=None,
                error=f"invalid parent PlacementScript: {exc}",
                metadata={"operator": self.name, "invalid_attempts": 1, "retries": 0},
            )

        valid_inspirations = []
        rejected_inspirations = 0
        for inspiration in inspirations:
            try:
                ps.Policy(inspiration.code)
            except ps.PlacementScriptError:
                rejected_inspirations += 1
                continue
            valid_inspirations.append(inspiration)

        invalid: list[dict[str, str | int]] = []
        operations = ("crossover", "subtree_mutation", "numeric_mutation")
        weights = (
            self.config.crossover_probability,
            self.config.subtree_mutation_probability,
            self.config.numeric_mutation_probability,
        )
        for attempt in range(1, self.config.max_attempts + 1):
            operation = rng.choices(operations, weights=weights, k=1)[0]
            try:
                child, detail = self._apply(
                    operation,
                    parent.code,
                    tuple(p.code for p in valid_inspirations),
                    rng,
                )
                if child == parent.code:
                    raise ValueError("operator produced an unchanged offspring")
                policy = ps.Policy(child)
            except (ps.PlacementScriptError, ValueError) as exc:
                invalid.append(
                    {
                        "attempt": attempt,
                        "operation": operation,
                        "error": str(exc),
                    }
                )
                continue
            return ProposalOffspring(
                source=child,
                metadata={
                    "operator": self.name,
                    "operation": operation,
                    "attempts": attempt,
                    "invalid_attempts": len(invalid),
                    "retries": attempt - 1,
                    "repair_calls": 0,
                    "rejected_inspirations": rejected_inspirations,
                    "parent_sha256": _digest(parent.code),
                    "offspring_sha256": _digest(child),
                    "policy_name": policy.name,
                    "policy_version": policy.version,
                    "detail": detail,
                    "invalid_history": invalid,
                    "api_costs": 0.0,
                },
            )

        return ProposalOffspring(
            source=None,
            error=f"no valid GP offspring after {self.config.max_attempts} attempts",
            metadata={
                "operator": self.name,
                "attempts": self.config.max_attempts,
                "invalid_attempts": len(invalid),
                "retries": max(0, self.config.max_attempts - 1),
                "repair_calls": 0,
                "rejected_inspirations": rejected_inspirations,
                "invalid_history": invalid,
                "api_costs": 0.0,
            },
        )

    def _apply(
        self,
        operation: str,
        parent: str,
        inspirations: tuple[str, ...],
        rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        if operation == "numeric_mutation":
            return self._numeric_mutation(parent, rng)
        if operation == "crossover":
            if not inspirations:
                raise ValueError("crossover needs at least one valid inspiration")
            donor_source = rng.choice(inspirations)
        else:
            donor_source = parent
        return self._tree_splice(parent, donor_source, rng, operation)

    @staticmethod
    def _tree_splice(
        parent: str,
        donor_source: str,
        rng: random.Random,
        operation: str,
    ) -> tuple[str, dict[str, Any]]:
        targets = _mutable_sites(parent)
        donors = _mutable_sites(donor_source)
        if not targets or not donors:
            raise ValueError("no mutable PlacementScript AST sites")
        target = rng.choice(targets)
        compatible = [site for site in donors if _compatible(target, site)]
        if donor_source == parent:
            compatible = [site for site in compatible if site["id"] != target["id"]]
        if not compatible:
            raise ValueError("no type-compatible donor site")
        donor = rng.choice(compatible)
        child = ps.crossover(parent, donor_source, target["id"], donor["id"])
        return child, {
            "kind": operation,
            "target": {
                "id": target["id"],
                "kind": target["kind"],
                "type": target["type"],
                "func": target["func"],
            },
            "donor": {
                "id": donor["id"],
                "kind": donor["kind"],
                "type": donor["type"],
                "func": donor["func"],
            },
            "donor_sha256": _digest(donor_source),
        }

    @staticmethod
    def _numeric_mutation(
        parent: str, rng: random.Random
    ) -> tuple[str, dict[str, Any]]:
        literals = [
            dict(literal)
            for literal in ps.enumerate_numeric_literals(parent)
            if not literal["is_const"]
        ]
        if not literals:
            raise ValueError("no mutable numeric literals")
        target = rng.choice(literals)
        if target["is_float"]:
            replacement = _replacement_float(float(target["text"]), rng)
        else:
            replacement = _replacement_int(int(target["text"]), rng)
        child = ps.patch_numeric_literals(parent, {target["id"]: replacement})
        return child, {
            "kind": "numeric_mutation",
            "literal_id": target["id"],
            "before": target["text"],
            "after": replacement,
        }


__all__ = ["GPOperatorConfig", "PlacementScriptGPOperator", "load_gp_operator_config"]
