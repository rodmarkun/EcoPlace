"""Shared proposal controls applied identically to both G-ablate arms."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import random
from collections.abc import Awaitable
from dataclasses import asdict, dataclass, replace
from typing import Any

import placementscript as ps
from shinka.core import ProposalOffspring
from shinka.core.proposal_operator import ProposalOperator, invoke_proposal_operator
from shinka.database import Program
from shinka.edit import apply_diff_patch, apply_numeric_patch


def _evolve_parts(source: str) -> tuple[str, str, str]:
    """Return immutable prefix, mutable body, and immutable suffix byte-for-byte."""
    lines = source.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if "EVOLVE-BLOCK-START" in line]
    ends = [index for index, line in enumerate(lines) if "EVOLVE-BLOCK-END" in line]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("policy must contain exactly one ordered EVOLVE-BLOCK pair")
    return (
        "".join(lines[: starts[0] + 1]),
        "".join(lines[starts[0] + 1 : ends[0]]),
        "".join(lines[ends[0] :]),
    )


def _evolve_boundaries(source: str) -> tuple[str, str]:
    """Return immutable prefix/suffix, including the unique marker lines."""
    prefix, _body, suffix = _evolve_parts(source)
    return prefix, suffix


def apply_scoped_diff(parent: str, patch: str) -> str:
    """Apply a Shinka diff strictly inside the mutable body without byte drift.

    Shinka's diff adapter normalizes trailing whitespace on the *whole* input before
    matching.  That is useful for fuzzy SEARCH context, but it also removes the final
    newline (and any immutable trailing spaces) from a PlacementScript policy.  The
    subsequent byte-for-byte scope check therefore rejected every otherwise-applied
    archived diff.  Isolating the mutable body preserves Shinka's anchoring,
    indentation correction, and sequential multi-hunk behavior while making it
    impossible for its normalization to touch the immutable shell.
    """
    prefix, body, suffix = _evolve_parts(parent)
    scoped_parent = f"# EVOLVE-BLOCK-START\n{body}# EVOLVE-BLOCK-END\n"
    result = apply_diff_patch(
        patch_str=patch,
        original_str=scoped_parent,
        language="python",
        verbose=False,
    )
    scoped_child, applied, _output, error, _patch_text, _patch_path = result
    if error is not None or applied < 1:
        raise ValueError(error or "diff completion changed no EVOLVE-BLOCK content")
    _scoped_prefix, child_body, _scoped_suffix = _evolve_parts(scoped_child)
    child = prefix + child_body + suffix
    if child == parent:
        raise ValueError("diff completion produced an unchanged policy")
    validate_scoped_edit(parent, child)
    return child


def validate_scoped_edit(parent: str, child: str) -> None:
    """Reject any offspring that changes bytes outside the EVOLVE-BLOCK."""
    parent_prefix, parent_suffix = _evolve_boundaries(parent)
    child_prefix, child_suffix = _evolve_boundaries(child)
    if child_prefix != parent_prefix or child_suffix != parent_suffix:
        raise ValueError("offspring changed immutable bytes outside EVOLVE-BLOCK")


ABLATION_KINDS = (
    "capacity_floor_extreme",
    "class_normalizer_removal",
    "constant_capacity",
    "feature_term_deletion",
    "guard_removal",
    "guard_threshold_zeroing",
    "ratio_denormalization",
    "raw_feature_substitution",
    "sentinel_scale_break",
    "state_feature_zeroing",
)


@dataclass(frozen=True)
class AblationSite:
    """One native PlacementScript AST mutation, with an auditable stable locator."""

    kind: str
    target_id: int
    function: str
    line: int
    node_kind: str
    node_type: str
    replacement_kind: str
    replacement: str
    donor_id: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ABLATION_KINDS:
            raise ValueError(f"unknown ablation kind {self.kind!r}")
        if self.replacement_kind not in {"subtree", "source"}:
            raise ValueError("unknown ablation replacement kind")
        if (self.replacement_kind == "subtree") is (self.donor_id is None):
            raise ValueError("subtree ablations require exactly one donor id")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utf8_offsets(source: str) -> list[int]:
    offsets = []
    position = 0
    for line in source.splitlines(keepends=True):
        offsets.append(position)
        position += len(line.encode("utf-8"))
    if not offsets:
        offsets.append(0)
    return offsets


def _node_span(node: ast.AST, offsets: list[int]) -> tuple[int, int] | None:
    if not all(
        hasattr(node, field)
        for field in ("lineno", "col_offset", "end_lineno", "end_col_offset")
    ):
        return None
    return (
        offsets[node.lineno - 1] + node.col_offset,
        offsets[node.end_lineno - 1] + node.end_col_offset,
    )


def _target_names(target: ast.AST) -> tuple[str, ...]:
    return tuple(node.id for node in ast.walk(target) if isinstance(node, ast.Name))


def _contains_capacity(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr.endswith("_capacity")
        for child in ast.walk(node)
    )


def _contains_nodes_len(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Name):
            continue
        if child.func.id != "len" or len(child.args) != 1:
            continue
        argument = child.args[0]
        if isinstance(argument, ast.Attribute) and argument.attr == "nodes":
            return True
    return False


def _native_ast_maps(source: str) -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, int], dict[str, Any]],
    ast.Module,
    list[int],
]:
    # Python's AST is used only to identify operators and field names. Site identity
    # and every rewrite are owned by the native PlacementScript AST API below.
    tree = ast.parse(source)
    sites = [dict(site) for site in ps.enumerate_subtrees(source)]
    by_span = {(site["start"], site["end"]): site for site in sites}
    return sites, by_span, tree, _utf8_offsets(source)


def enumerate_ablation_sites(source: str) -> tuple[AblationSite, ...]:
    """Enumerate semantic-shock candidates from typed native AST sites.

    No source patching or substring replacement is used. The Python AST supplies
    operator/field semantics missing from ``enumerate_subtrees``; exact target and
    donor identities come from PlacementScript's typed AST and mutations go through
    ``crossover``/``replace_subtree``.
    """
    ps.Policy(source)
    sites, by_span, tree, offsets = _native_ast_maps(source)
    native_by_id = {site["id"]: site for site in sites}
    ast_parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def native(node: ast.AST) -> dict[str, Any] | None:
        span = _node_span(node, offsets)
        return None if span is None else by_span.get(span)

    def add(
        kind: str,
        target: dict[str, Any] | None,
        *,
        replacement: str,
        replacement_kind: str = "source",
        donor: dict[str, Any] | None = None,
        detail: str,
    ) -> None:
        if target is None or target["func"] is None:
            return
        candidates.append(
            AblationSite(
                kind=kind,
                target_id=target["id"],
                function=target["func"],
                line=target["line"],
                node_kind=target["kind"],
                node_type=str(target["type"]),
                replacement_kind=replacement_kind,
                replacement=replacement,
                donor_id=None if donor is None else donor["id"],
                detail=detail,
            )
        )

    candidates: list[AblationSite] = []
    function_nodes = {
        node: node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    def enclosing_function(node: ast.AST) -> str | None:
        current: ast.AST | None = node
        while current is not None:
            if current in function_nodes:
                return function_nodes[current]
            current = ast_parents.get(current)
        return None

    # Delete either side of arithmetic feature combinations. For additive score
    # aggregates, also offer earlier in-scope raw float features; this is the native
    # one-splice route that can produce ``cpu_free / 2.0`` from the seed.
    all_names = [node for node in ast.walk(tree) if isinstance(node, ast.Name)]
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        target = native(node)
        left = native(node.left)
        right = native(node.right)
        if target is None:
            continue
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            for side, donor in (("left", left), ("right", right)):
                if donor is not None and donor["type"] == target["type"]:
                    add(
                        "feature_term_deletion",
                        target,
                        replacement=f"native-subtree:{donor['id']}",
                        replacement_kind="subtree",
                        donor=donor,
                        detail=f"retain-{side}-operand",
                    )
            if isinstance(node.op, ast.Add):
                func = enclosing_function(node)
                for name_node in all_names:
                    donor = native(name_node)
                    if (
                        donor is None
                        or donor["func"] != func
                        or donor["line"] >= target["line"]
                        or donor["type"] != target["type"]
                        or donor["parent"] == target["id"]
                    ):
                        continue
                    add(
                        "raw_feature_substitution",
                        target,
                        replacement=f"native-subtree:{donor['id']}",
                        replacement_kind="subtree",
                        donor=donor,
                        detail=f"replace-aggregate-with-raw-name:{name_node.id}",
                    )
        elif isinstance(node.op, ast.Div) and left is not None:
            kind = (
                "ratio_denormalization"
                if _contains_capacity(node.right)
                or (
                    isinstance(node.right, ast.Name)
                    and ("cap" in node.right.id or "capacity" in node.right.id)
                )
                else "class_normalizer_removal"
            )
            add(
                kind,
                target,
                replacement=f"native-subtree:{left['id']}",
                replacement_kind="subtree",
                donor=left,
                detail="drop-division-retain-numerator",
            )

    # Capacity references and capacity-derived assignments can be replaced with
    # explicit domain-shock constants. PlacementScript re-type-checks each result.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            names = _target_names(node.targets[0])
            target = native(node.value)
            if (
                target is not None
                and any("cap" in name or "capacity" in name for name in names)
                and _contains_capacity(node.value)
            ):
                constants = ("0.0", "2.0", "16.0") if target["type"] == "float" else (
                    "0",
                    "2",
                    "16",
                )
                for constant in constants:
                    add(
                        "constant_capacity",
                        target,
                        replacement=constant,
                        detail=f"capacity-assignment-to:{constant}",
                    )

            sentinel = next(
                (name for name in names if name in {"hi", "lo", "hi_u", "lo_u"}),
                None,
            )
            if sentinel is not None:
                for child in ast.walk(node.value):
                    if not isinstance(child, ast.Constant) or isinstance(child.value, bool):
                        continue
                    literal = native(child)
                    if literal is None or literal["kind"] not in {"int_lit", "float_lit"}:
                        continue
                    replacement = "16.0" if literal["type"] == "float" else "16"
                    add(
                        "sentinel_scale_break",
                        literal,
                        replacement=replacement,
                        detail=f"{sentinel}-magnitude-to-16",
                    )

        if isinstance(node, ast.Attribute) and node.attr.endswith("_allocated"):
            target = native(node)
            if target is not None:
                replacement = "0.0" if target["type"] == "float" else "0"
                add(
                    "state_feature_zeroing",
                    target,
                    replacement=replacement,
                    detail=f"zero-state-field:{node.attr}",
                )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in {"max", "min"} or not _contains_capacity(node):
                continue
            for argument in node.args:
                if not isinstance(argument, ast.Constant) or isinstance(argument.value, bool):
                    continue
                target = native(argument)
                if target is None:
                    continue
                replacement = "16.0" if target["type"] == "float" else "16"
                add(
                    "capacity_floor_extreme",
                    target,
                    replacement=replacement,
                    detail=f"{node.func.id}-capacity-floor-to-16",
                )

        if isinstance(node, ast.If):
            condition = native(node.test)
            if condition is not None:
                early_exit = any(isinstance(child, ast.Return) for child in node.body)
                replacement = "False" if early_exit else "True"
                add(
                    "guard_removal",
                    condition,
                    replacement=replacement,
                    detail=(
                        "disable-early-exit-guard"
                        if early_exit
                        else "make-wrapping-guard-unconditional"
                    ),
                )
            if not _contains_nodes_len(node.test):
                continue
            for child in ast.walk(node.test):
                if not isinstance(child, ast.Constant) or isinstance(child.value, bool):
                    continue
                if child.value == 0:
                    continue
                target = native(child)
                if target is None or target["kind"] not in {"int_lit", "float_lit"}:
                    continue
                replacement = "0.0" if target["type"] == "float" else "0"
                add(
                    "guard_threshold_zeroing",
                    target,
                    replacement=replacement,
                    detail="zero-node-class-count-threshold",
                )

    unique = {
        (
            candidate.kind,
            candidate.target_id,
            candidate.replacement_kind,
            candidate.replacement,
            candidate.donor_id,
        ): candidate
        for candidate in candidates
        if candidate.target_id in native_by_id
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda site: (
                site.kind,
                site.function,
                site.line,
                site.target_id,
                site.replacement,
                -1 if site.donor_id is None else site.donor_id,
            ),
        )
    )


class AblationProbeOperator:
    """Seeded LLM-free semantic-shock proposer over native AST sites."""

    name = "ablation_probe"

    def __init__(self, *, max_attempts: int = 24) -> None:
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("ablation max_attempts must be a positive integer")
        self.max_attempts = max_attempts

    @staticmethod
    def _apply(source: str, site: AblationSite) -> str:
        if site.replacement_kind == "subtree":
            return ps.crossover(source, source, site.target_id, site.donor_id)
        return ps.replace_subtree(source, site.target_id, site.replacement)

    def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("ablation proposer requires an explicit random.Random")
        common = {
            "operator": self.name,
            "api_costs_microusd": 0,
            "inspirations_offered": len(inspirations),
            "inspirations_consumed": 0,
            "feedback_consumed": False,
        }
        try:
            sites = list(enumerate_ablation_sites(parent.code))
        except (SyntaxError, ValueError, ps.PlacementScriptError) as exc:
            return ProposalOffspring(
                source=None,
                error=f"invalid parent PlacementScript: {exc}",
                metadata={**common, "enumerated_site_count": 0, "invalid_attempts": 1},
            )
        rng.shuffle(sites)
        failures = []
        for attempt, site in enumerate(sites[: self.max_attempts], start=1):
            try:
                child = self._apply(parent.code, site)
                if child == parent.code:
                    raise ValueError("ablation produced unchanged source")
                validate_scoped_edit(parent.code, child)
                policy = ps.Policy(child)
            except (ValueError, ps.PlacementScriptError) as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "ablation_kind": site.kind,
                        "site": site.to_dict(),
                        "error": str(exc),
                    }
                )
                continue
            return ProposalOffspring(
                source=child,
                metadata={
                    **common,
                    "ablation_kind": site.kind,
                    "ablation_site": site.to_dict(),
                    "enumerated_site_count": len(sites),
                    "attempts": attempt,
                    "invalid_attempts": len(failures),
                    "invalid_history": failures,
                    "parent_sha256": hashlib.sha256(parent.code.encode()).hexdigest(),
                    "offspring_sha256": hashlib.sha256(child.encode()).hexdigest(),
                    "policy_name": policy.name,
                    "policy_version": policy.version,
                },
            )
        return ProposalOffspring(
            source=None,
            error=(
                "no loadable ablation offspring after "
                f"{min(len(sites), self.max_attempts)} attempts"
            ),
            metadata={
                **common,
                "enumerated_site_count": len(sites),
                "attempts": min(len(sites), self.max_attempts),
                "invalid_attempts": len(failures),
                "invalid_history": failures,
            },
        )


class AblationProbeInterleaveOperator:
    """Add the same ablation route to any PlacementScript proposal arm."""

    def __init__(
        self,
        primary: ProposalOperator,
        *,
        probability: float,
        shared_numeric_probability: float = 0.0,
        probe: AblationProbeOperator | None = None,
    ) -> None:
        for name, value in {
            "ablation probe probability": probability,
            "shared numeric probability": shared_numeric_probability,
        }.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if probability <= 0.0:
            raise ValueError("ablation interleave requires a positive route share")
        if probability + shared_numeric_probability > 1.0:
            raise ValueError("ablation and numeric route shares must sum to at most 1")
        self.primary = primary
        self.probability = float(probability)
        self.shared_numeric_probability = float(shared_numeric_probability)
        self.probe = probe or AblationProbeOperator()
        self.name = f"ablation_probe_interleave::{primary.name}"

    @property
    def cost_ledger(self):
        return getattr(self.primary, "cost_ledger", None)

    @property
    def usage_ledger(self):
        return getattr(self.primary, "usage_ledger", None)

    @property
    def engine_descriptor(self):
        return getattr(self.primary, "engine_descriptor", None)

    def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring | Awaitable[ProposalOffspring]:
        if not isinstance(rng, random.Random):
            raise TypeError("ablation interleave requires an explicit random.Random")
        proposal = self.propose_seeded(
            parent,
            inspirations,
            route_seed=rng.randrange(0, 2**63),
            ablation_seed=rng.randrange(0, 2**63),
            numeric_seed=rng.randrange(0, 2**63),
            primary_seed=rng.randrange(0, 2**63),
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(proposal)
        return proposal

    async def propose_seeded(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        *,
        route_seed: int,
        ablation_seed: int,
        numeric_seed: int,
        primary_seed: int,
        primary_mode: str | None = None,
        primary_repair: Any | None = None,
    ) -> ProposalOffspring:
        for name, seed in {
            "route_seed": route_seed,
            "ablation_seed": ablation_seed,
            "numeric_seed": numeric_seed,
            "primary_seed": primary_seed,
        }.items():
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
                raise ValueError(f"{name} must be an explicit integer in [0, 2**63)")
        route_draw = random.Random(route_seed).random()
        common = {
            "interleave_operator": self.name,
            "primary_operator": self.primary.name,
            "ablation_probe_probability": self.probability,
            "shared_numeric_probability": self.shared_numeric_probability,
            "shared_route_draw": route_draw,
            "shared_route_seed": route_seed,
            "ablation_probe_seed": ablation_seed,
            "shared_numeric_seed": numeric_seed,
            "arm_primary_seed": primary_seed,
        }
        if route_draw < self.probability:
            result = self.probe(parent, inspirations, random.Random(ablation_seed))
            return replace(
                result,
                metadata={**result.metadata, **common, "route": "ablation_probe"},
            )
        if route_draw < self.probability + self.shared_numeric_probability:
            source, applied, _output, error, notes, _patch = apply_numeric_patch(
                original_str=parent.code,
                language="python",
                verbose=False,
                seed=numeric_seed,
            )
            metadata = {
                **common,
                "route": "shared_numeric",
                "numeric_patch_notes": notes,
                "api_costs_microusd": 0,
            }
            if error is not None or applied < 1:
                return ProposalOffspring(
                    source=None,
                    error=error or "shared numeric patch changed no literals",
                    metadata=metadata,
                )
            try:
                validate_scoped_edit(parent.code, source)
                policy = ps.Policy(source)
            except (ValueError, ps.PlacementScriptError) as exc:
                return ProposalOffspring(source=None, error=str(exc), metadata=metadata)
            metadata.update(
                {
                    "offspring_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "policy_name": policy.name,
                    "policy_version": policy.version,
                }
            )
            return ProposalOffspring(source=source, metadata=metadata)

        repair_method = getattr(self.primary, "propose_with_repair", None)
        if primary_mode is not None or primary_repair is not None:
            if not callable(repair_method):
                raise ValueError("primary operator does not support grammar-aware repair")
            result = repair_method(
                parent,
                inspirations,
                random.Random(primary_seed),
                mode=primary_mode,
                repair=primary_repair,
            )
        else:
            result = await invoke_proposal_operator(
                self.primary,
                parent,
                inspirations,
                random.Random(primary_seed),
            )
        return replace(result, metadata={**result.metadata, **common, "route": "primary"})


class SharedNumericInterleaveOperator:
    """Interleave Shinka's seeded local numeric patch before either primary arm."""

    def __init__(self, primary: ProposalOperator, *, probability: float) -> None:
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise TypeError("shared numeric probability must be numeric")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("shared numeric probability must be in [0, 1]")
        self.primary = primary
        self.probability = float(probability)
        self.name = f"shared_numeric_interleave::{primary.name}"

    @property
    def cost_ledger(self):
        return getattr(self.primary, "cost_ledger", None)

    @property
    def usage_ledger(self):
        return getattr(self.primary, "usage_ledger", None)

    @property
    def engine_descriptor(self):
        return getattr(self.primary, "engine_descriptor", None)

    async def __call__(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        rng: random.Random,
    ) -> ProposalOffspring:
        if not isinstance(rng, random.Random):
            raise TypeError("shared proposer requires the runner-owned random.Random")
        # Generic Shinka callers provide one dispatcher RNG. It is used only as a
        # deterministic seed splitter; no proposal component shares another's stream.
        route_seed = rng.randrange(0, 2**63)
        numeric_seed = rng.randrange(0, 2**63)
        primary_seed = rng.randrange(0, 2**63)
        return await self.propose_seeded(
            parent,
            inspirations,
            route_seed=route_seed,
            numeric_seed=numeric_seed,
            primary_seed=primary_seed,
        )

    async def propose_seeded(
        self,
        parent: Program,
        inspirations: tuple[Program, ...],
        *,
        route_seed: int,
        numeric_seed: int,
        primary_seed: int,
        primary_mode: str | None = None,
        primary_repair: Any | None = None,
    ) -> ProposalOffspring:
        for name, seed in {
            "route_seed": route_seed,
            "numeric_seed": numeric_seed,
            "primary_seed": primary_seed,
        }.items():
            if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
                raise ValueError(f"{name} must be an explicit integer in [0, 2**63)")
        route_draw = random.Random(route_seed).random()
        common = {
            "interleave_operator": self.name,
            "primary_operator": self.primary.name,
            "shared_numeric_probability": self.probability,
            "shared_route_draw": route_draw,
            "shared_route_seed": route_seed,
            "shared_numeric_seed": numeric_seed,
            "arm_primary_seed": primary_seed,
        }
        if route_draw >= self.probability:
            repair_method = getattr(self.primary, "propose_with_repair", None)
            if primary_mode is not None or primary_repair is not None:
                if not callable(repair_method):
                    raise ValueError("primary operator does not support grammar-aware repair")
                result = repair_method(
                    parent,
                    inspirations,
                    random.Random(primary_seed),
                    mode=primary_mode,
                    repair=primary_repair,
                )
            else:
                result = await invoke_proposal_operator(
                    self.primary,
                    parent,
                    inspirations,
                    random.Random(primary_seed),
                )
            return replace(result, metadata={**result.metadata, **common, "route": "primary"})

        source, applied, _output, error, notes, _patch = apply_numeric_patch(
            original_str=parent.code,
            language="python",
            verbose=False,
            seed=numeric_seed,
        )
        metadata = {
            **common,
            "route": "shared_numeric",
            "numeric_patch_notes": notes,
            "api_costs_microusd": 0,
        }
        if error is not None or applied < 1:
            return ProposalOffspring(
                source=None,
                error=error or "shared numeric patch changed no literals",
                metadata=metadata,
            )
        try:
            validate_scoped_edit(parent.code, source)
            policy = ps.Policy(source)
        except (ValueError, ps.PlacementScriptError) as exc:
            return ProposalOffspring(source=None, error=str(exc), metadata=metadata)
        metadata.update(
            {
                "offspring_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "policy_name": policy.name,
                "policy_version": policy.version,
            }
        )
        return ProposalOffspring(source=source, metadata=metadata)


__all__ = [
    "ABLATION_KINDS",
    "AblationProbeInterleaveOperator",
    "AblationProbeOperator",
    "AblationSite",
    "SharedNumericInterleaveOperator",
    "apply_scoped_diff",
    "enumerate_ablation_sites",
    "validate_scoped_edit",
]
