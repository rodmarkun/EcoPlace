"""Classic strongly typed GP hyper-heuristic baseline (TODO 7.3).

This is deliberately *not* the same-DSL ablation.  Individuals are DEAP
``PrimitiveTree`` expressions over a conventional fixed primitive/feature set and emit
one scalar placement priority.  Every candidate is wrapped as the common
``PolicyBundleAdapter`` and traverses the registered L0/L1/L2 ladder.  Generation,
selection, crossover, and mutation consume only an explicit root seed; DEAP's module
global RNG is never called.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Protocol

from deap import gp

from baselines.adapter import PolicyBundleAdapter
from evolution.campaign import EvaluatorWorkBudget
from evolution.evaluation import TaskEvaluationConfig
from evolution.ladder import LadderConfig, LadderResult
from sim.actuation import CallableProposer

FEATURE_NAMES = (
    "cpu_used_after",
    "mem_used_after",
    "cpu_free_after",
    "mem_free_after",
    "pod_cpu_fraction",
    "pod_mem_fraction",
    "marginal_power",
    "idle_power_per_cpu",
    "tier",
    "suspended",
    "neighbor_latency",
    "neighbor_traffic",
)
_CLIP = 1_000_000.0
# Koza's height cap, and DEAP's ``gp.staticLimit`` default. Initial trees are built to
# ``GPHHConfig.max_depth`` (4); this bounds what variation may grow them into.
DEFAULT_MAX_TREE_HEIGHT = 17


def _finite(value: float) -> float:
    if not math.isfinite(value):
        return math.copysign(_CLIP, value) if not math.isnan(value) else 0.0
    return min(max(value, -_CLIP), _CLIP)


def _add(a: float, b: float) -> float:
    return _finite(a + b)


def _sub(a: float, b: float) -> float:
    return _finite(a - b)


def _mul(a: float, b: float) -> float:
    return _finite(a * b)


def _div(a: float, b: float) -> float:
    return _finite(a if abs(b) < 1e-12 else a / b)


def _neg(a: float) -> float:
    return _finite(-a)


def _abs(a: float) -> float:
    return abs(_finite(a))


def _minimum(a: float, b: float) -> float:
    return min(a, b)


def _maximum(a: float, b: float) -> float:
    return max(a, b)


def _less(a: float, b: float) -> bool:
    return a < b


def _greater(a: float, b: float) -> bool:
    return a > b


def _and(a: bool, b: bool) -> bool:
    return a and b


def _or(a: bool, b: bool) -> bool:
    return a or b


def _not(a: bool) -> bool:
    return not a


def _if(condition: bool, yes: float, no: float) -> float:
    return yes if condition else no


def primitive_set() -> gp.PrimitiveSetTyped:
    """Build a fresh strongly typed DEAP primitive set (float + bool)."""
    pset = gp.PrimitiveSetTyped("GPHH", [float] * len(FEATURE_NAMES), float)
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(FEATURE_NAMES)})
    for fn, args, ret, name in (
        (_add, (float, float), float, "add"),
        (_sub, (float, float), float, "sub"),
        (_mul, (float, float), float, "mul"),
        (_div, (float, float), float, "pdiv"),
        (_neg, (float,), float, "neg"),
        (_abs, (float,), float, "abs"),
        (_minimum, (float, float), float, "min"),
        (_maximum, (float, float), float, "max"),
        (_less, (float, float), bool, "lt"),
        (_greater, (float, float), bool, "gt"),
        (_and, (bool, bool), bool, "and_"),
        (_or, (bool, bool), bool, "or_"),
        (_not, (bool,), bool, "not_"),
        (_if, (bool, float, float), float, "if_then_else"),
    ):
        pset.addPrimitive(fn, list(args), ret, name=name)
    for constant in (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        pset.addTerminal(constant, float)
    return pset


def _generate_subtree(
    pset: gp.PrimitiveSetTyped,
    rng: random.Random,
    return_type: type,
    *,
    depth: int,
    min_depth: int,
    max_depth: int,
) -> list[Any]:
    terminals = list(pset.terminals[return_type])
    primitives = list(pset.primitives[return_type])
    choose_terminal = bool(terminals) and (
        depth >= max_depth or (depth >= min_depth and rng.random() < 0.35)
    )
    if choose_terminal or not primitives:
        if not terminals:
            raise ValueError(f"primitive set cannot terminate type {return_type.__name__}")
        return [rng.choice(terminals)]
    if depth >= max_depth:
        # `max_depth` was only ever enforced through the terminal branch, and `bool` has
        # no terminals in this primitive set: `and_`, `or_` and `not_` all take booleans,
        # so a boolean subtree could only stop by drawing `gt`/`lt`. That is a branching
        # process with no bound, and it is what actually produced the trees Python
        # refused to parse -- not variation. Past `max_depth`, restrict the draw to
        # primitives whose arguments can all terminate immediately (for `bool`, exactly
        # the comparisons), so every branch closes on the next level.
        closing = [
            candidate
            for candidate in primitives
            if all(pset.terminals[argument] for argument in candidate.args)
        ]
        if closing:
            primitives = closing
    primitive = rng.choice(primitives)
    expression: list[Any] = [primitive]
    for argument_type in primitive.args:
        expression.extend(
            _generate_subtree(
                pset,
                rng,
                argument_type,
                depth=depth + 1,
                min_depth=min_depth,
                max_depth=max_depth,
            )
        )
    return expression


def generate_tree(
    pset: gp.PrimitiveSetTyped,
    rng: random.Random,
    *,
    min_depth: int,
    max_depth: int,
) -> gp.PrimitiveTree:
    if not isinstance(rng, random.Random):
        raise TypeError("GPHH tree generation requires an explicit random.Random")
    return gp.PrimitiveTree(
        _generate_subtree(
            pset,
            rng,
            pset.ret,
            depth=0,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    )


def mutate_tree(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped,
    rng: random.Random,
    *,
    max_depth: int,
    max_height: int = DEFAULT_MAX_TREE_HEIGHT,
) -> gp.PrimitiveTree:
    """Replace one subtree, rejecting the child if it would exceed ``max_height``.

    Subtree mutation grafts a fresh subtree of up to ``max_depth`` at an arbitrary point,
    so an unbounded child can be far deeper than its parent. Without the bound the
    population bloats until Python's own parser refuses the compiled lambda
    ("SyntaxError: too many nested parentheses"), which killed a campaign outright once
    the search ran long enough to get there. Koza's height cap, and what DEAP's
    ``gp.staticLimit`` does: on violation, return the parent unchanged.
    """
    child = gp.PrimitiveTree(tree)
    index = rng.randrange(len(child))
    return_type = child[index].ret
    replacement = _generate_subtree(
        pset,
        rng,
        return_type,
        depth=0,
        min_depth=0,
        max_depth=max_depth,
    )
    child[child.searchSubtree(index)] = replacement
    if child.height > max_height:
        return gp.PrimitiveTree(tree)
    return child


def crossover_trees(
    left: gp.PrimitiveTree,
    right: gp.PrimitiveTree,
    rng: random.Random,
    *,
    max_height: int = DEFAULT_MAX_TREE_HEIGHT,
) -> tuple[gp.PrimitiveTree, gp.PrimitiveTree]:
    """Swap one compatible subtree pair, rejecting children that exceed ``max_height``.

    Subtree crossover is the other half of GP bloat: swapping a deep subtree into a
    shallow site grows the child without bound. Same rule as :func:`mutate_tree` -- an
    over-height child is discarded and its parent returned, per subtree.
    """
    a, b = gp.PrimitiveTree(left), gp.PrimitiveTree(right)
    compatible = [
        (i, j)
        for i, left_node in enumerate(a)
        for j, right_node in enumerate(b)
        if left_node.ret is right_node.ret
    ]
    if not compatible:
        return a, b
    i, j = rng.choice(compatible)
    a_slice, b_slice = a.searchSubtree(i), b.searchSubtree(j)
    a_part, b_part = list(a[a_slice]), list(b[b_slice])
    a[a_slice], b[b_slice] = b_part, a_part
    if a.height > max_height:
        a = gp.PrimitiveTree(left)
    if b.height > max_height:
        b = gp.PrimitiveTree(right)
    return a, b


def _features(pod: dict, node: dict, ctx: dict) -> tuple[float, ...]:
    cpu_capacity = max(node["cpu_capacity"], 1)
    mem_capacity = max(node["mem_capacity"], 1)
    cpu_after = min((node["cpu_allocated"] + pod["cpu_request"]) / cpu_capacity, 1.0)
    mem_after = min((node["mem_allocated"] + pod["mem_request"]) / mem_capacity, 1.0)
    latency = 0.0
    traffic = 0.0
    graph = ctx["graph"]
    for src, dst, rate, payload in zip(
        graph["src"], graph["dst"], graph["rate"], graph["payload_kb"], strict=True
    ):
        other = dst if src == pod["id"] else src if dst == pod["id"] else -1
        if other < 0 or other == pod["id"]:
            continue
        placed = ctx["pods"][other]["current_node"]
        if placed >= 0:
            link = node["latency_to_ms"][placed]
            latency += rate * link
            traffic += rate * payload * link
    power_span = max(node["power_max_w"] - node["power_idle_w"], 0.0)
    return (
        cpu_after,
        mem_after,
        1.0 - cpu_after,
        1.0 - mem_after,
        pod["cpu_request"] / cpu_capacity,
        pod["mem_request"] / mem_capacity,
        power_span * pod["cpu_request"] / cpu_capacity,
        node["power_idle_w"] / cpu_capacity,
        float(node["tier"]),
        1.0 if node["suspended"] else 0.0,
        latency / 1000.0,
        traffic / 1_000_000.0,
    )


class GPHHScorer:
    def __init__(self, tree: gp.PrimitiveTree, pset: gp.PrimitiveSetTyped | None = None) -> None:
        self.tree = gp.PrimitiveTree(tree)
        self.pset = pset or primitive_set()
        self.expression = str(self.tree)
        self.sha256 = hashlib.sha256(self.expression.encode()).hexdigest()
        self.name = f"gphh:{self.sha256[:12]}"
        self._fn = gp.compile(self.tree, self.pset)

    def score_batch(self, pod: dict, nodes: list[dict], ctx: dict) -> list[float]:
        return [_finite(float(self._fn(*_features(pod, node, ctx)))) for node in nodes]


class GPHHProposer:
    """Use the same rule to propose improving moves under the fixed gateway budget."""

    def __init__(self, scorer: GPHHScorer) -> None:
        self.scorer = scorer
        self.name = scorer.name

    def propose_moves(self, state: dict, graph: dict, budget: int) -> list[dict[str, int]]:
        if budget <= 0:
            return []
        nodes = copy.deepcopy(state["nodes"])
        pods = copy.deepcopy(state["pods"])
        by_id = {node["id"]: node for node in nodes}
        ctx = {"nodes": nodes, "pods": pods, "graph": copy.deepcopy(graph), "now_ms": 0}
        out = []
        for pod in pods:
            if len(out) >= budget:
                break
            current = pod["current_node"]
            if current < 0 or pod["stateful"] or pod["has_pdb"]:
                continue
            by_id[current]["cpu_allocated"] -= pod["cpu_request"]
            by_id[current]["mem_allocated"] -= pod["mem_request"]
            feasible = [
                node
                for node in nodes
                if not node["suspended"]
                and node["cpu_allocated"] + pod["cpu_request"] <= node["cpu_capacity"]
                and node["mem_allocated"] + pod["mem_request"] <= node["mem_capacity"]
            ]
            feasible_ids = [node["id"] for node in feasible]
            if current not in feasible_ids:
                by_id[current]["cpu_allocated"] += pod["cpu_request"]
                by_id[current]["mem_allocated"] += pod["mem_request"]
                continue
            scores = self.scorer.score_batch(pod, feasible, ctx)
            best_index = max(
                range(len(feasible)),
                key=lambda i: (scores[i], -feasible[i]["id"]),
            )
            best = feasible[best_index]
            current_score = scores[feasible_ids.index(current)]
            if best["id"] != current and scores[best_index] > current_score:
                out.append({"kind": 0, "pod": pod["id"], "dest": best["id"]})
                pod["current_node"] = best["id"]
                best["cpu_allocated"] += pod["cpu_request"]
                best["mem_allocated"] += pod["mem_request"]
            else:
                by_id[current]["cpu_allocated"] += pod["cpu_request"]
                by_id[current]["mem_allocated"] += pod["mem_request"]
        return out


def gphh_bundle(
    tree: gp.PrimitiveTree,
    pset: gp.PrimitiveSetTyped | None = None,
) -> PolicyBundleAdapter:
    scorer = GPHHScorer(tree, pset)
    return PolicyBundleAdapter(
        name="classic_gphh",
        version=f"tree-{scorer.sha256[:12]}",
        scorer=scorer,
        proposer=CallableProposer("classic_gphh", GPHHProposer(scorer).propose_moves),
        representation="deap-strongly-typed-primitive-tree",
        artifact_sha256=scorer.sha256,
        metadata={
            "status": "confirmatory-reusable-policy",
            "deap_representation": "gp.PrimitiveTree",
            "tree": scorer.expression,
            "tree_sha256": scorer.sha256,
            "feature_names": list(FEATURE_NAMES),
            "return_type": "float",
            "internal_types": ["float", "bool"],
        },
    )


@dataclass(frozen=True)
class GPHHConfig:
    population_size: int = 16
    generations: int = 8
    elite_size: int = 2
    tournament_size: int = 3
    min_depth: int = 1
    max_depth: int = 4
    max_tree_height: int = DEFAULT_MAX_TREE_HEIGHT
    crossover_probability: float = 0.7
    mutation_probability: float = 0.25

    def __post_init__(self) -> None:
        for name in ("population_size", "generations", "elite_size", "tournament_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if self.elite_size > self.population_size:
            raise ValueError("elite_size cannot exceed population_size")
        if not 0 <= self.min_depth <= self.max_depth:
            raise ValueError("need 0 <= min_depth <= max_depth")
        if (
            not isinstance(self.max_tree_height, int)
            or isinstance(self.max_tree_height, bool)
            or self.max_tree_height < self.max_depth
        ):
            raise ValueError("max_tree_height must be an int no smaller than max_depth")
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")


class EvaluateGPHH(Protocol):
    def __call__(
        self,
        bundle: PolicyBundleAdapter,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        proposal_seq: int,
    ) -> LadderResult: ...


@dataclass(frozen=True)
class GPHHCandidate:
    proposal_seq: int
    generation: int
    tree: gp.PrimitiveTree
    result: LadderResult
    candidate_microwork: int
    cumulative_microwork: int

    @property
    def selection_key(self) -> tuple[Any, ...]:
        if self.result.promoted_to_l2 and self.result.evaluation is not None:
            return (0, *self.result.evaluation.fitness.sort_key())
        return (1, self.proposal_seq)

    def to_dict(self) -> dict[str, Any]:
        expression = str(self.tree)
        return {
            "proposal_seq": self.proposal_seq,
            "generation": self.generation,
            "tree": expression,
            "tree_sha256": hashlib.sha256(expression.encode()).hexdigest(),
            "candidate_microwork": self.candidate_microwork,
            "cumulative_microwork": self.cumulative_microwork,
            "selection_key": list(self.selection_key),
            "ladder": self.result.to_dict(),
        }


@dataclass(frozen=True)
class GPHHRun:
    root_seed: int
    config: GPHHConfig
    candidates: tuple[GPHHCandidate, ...]
    champion: GPHHCandidate | None
    spent_microwork: int
    stop_reason: str
    population_proposal_seqs_by_generation: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-classic-gphh-run-v2",
            "root_seed": self.root_seed,
            "config": dataclasses.asdict(self.config),
            "spent_microwork": self.spent_microwork,
            "stop_reason": self.stop_reason,
            "population_proposal_seqs_by_generation": [
                list(generation) for generation in self.population_proposal_seqs_by_generation
            ],
            "champion_proposal_seq": (
                None if self.champion is None else self.champion.proposal_seq
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {**payload, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _default_evaluate(
    bundle: PolicyBundleAdapter,
    task: TaskEvaluationConfig,
    ladder: LadderConfig,
    proposal_seq: int,
) -> LadderResult:
    return bundle.evaluate_through_ladder(task, ladder, proposal_seq=proposal_seq)


class ClassicGPHH:
    def __init__(
        self,
        config: GPHHConfig | None = None,
        *,
        evaluate_fn: EvaluateGPHH = _default_evaluate,
    ) -> None:
        self.config = config or GPHHConfig()
        self.evaluate_fn = evaluate_fn
        self.pset = primitive_set()

    @staticmethod
    def _tournament(
        population: list[GPHHCandidate], size: int, rng: random.Random
    ) -> GPHHCandidate:
        competitors = [rng.choice(population) for _ in range(size)]
        return min(competitors, key=lambda candidate: candidate.selection_key)

    def run(
        self,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        work: EvaluatorWorkBudget,
        *,
        root_seed: int,
        initial_trees: tuple[gp.PrimitiveTree, ...] = (),
    ) -> GPHHRun:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        rng = random.Random(root_seed)
        candidates: list[GPHHCandidate] = []
        population: list[GPHHCandidate] = []
        spent = 0
        proposal_seq = 0

        if len(initial_trees) > self.config.population_size:
            raise ValueError("initial_trees cannot exceed the GPHH population size")
        seeded_trees = []
        for tree in initial_trees:
            copied = gp.PrimitiveTree(tree)
            gp.compile(copied, self.pset)
            seeded_trees.append(copied)

        def evaluate(tree: gp.PrimitiveTree, generation: int) -> GPHHCandidate | None:
            nonlocal proposal_seq, spent
            if spent + work.max_candidate_microwork > work.max_microwork:
                return None
            bundle = gphh_bundle(tree, self.pset)
            result = self.evaluate_fn(bundle, task, ladder, proposal_seq)
            if (
                result.policy_name != bundle.name
                or result.policy_version != bundle.version
                or result.policy_identity_sha256 != bundle.identity_sha256
                or result.policy_artifact_sha256 != bundle.artifact_sha256
            ):
                raise ValueError("GPHH evaluator returned a mismatched policy identity")
            consumed = work.work_for_trace(result.trace)
            if consumed > work.max_candidate_microwork:
                raise ValueError("ladder trace exceeds registered per-candidate work")
            spent += consumed
            # The run keeps every candidate until it returns; the harness drops
            # ``instances[].episode`` when it writes arm results, so the full episode
            # ledgers held here were never used. Retaining them cost 60-90 MB per
            # candidate and pushed 224.0-work roots into swap (2026-09-04).
            result = result.without_episode_ledgers()
            candidate = GPHHCandidate(
                proposal_seq=proposal_seq,
                generation=generation,
                tree=gp.PrimitiveTree(tree),
                result=result,
                candidate_microwork=consumed,
                cumulative_microwork=spent,
            )
            proposal_seq += 1
            candidates.append(candidate)
            return candidate

        for tree in seeded_trees:
            candidate = evaluate(tree, 0)
            if candidate is None:
                break
            population.append(candidate)

        for _ in range(self.config.population_size - len(population)):
            candidate = evaluate(
                generate_tree(
                    self.pset,
                    rng,
                    min_depth=self.config.min_depth,
                    max_depth=self.config.max_depth,
                ),
                0,
            )
            if candidate is None:
                break
            population.append(candidate)

        stop_reason = "generation-limit"
        population_history = [tuple(candidate.proposal_seq for candidate in population)]
        for generation in range(1, self.config.generations + 1):
            if not population:
                stop_reason = "normalized-work-budget-exhausted"
                break
            ranked = sorted(population, key=lambda candidate: candidate.selection_key)
            elites = ranked[: self.config.elite_size]
            offspring: list[GPHHCandidate] = []
            offspring_target = self.config.population_size - len(elites)
            while len(offspring) < offspring_target:
                parent_a = self._tournament(population, self.config.tournament_size, rng)
                tree = gp.PrimitiveTree(parent_a.tree)
                if rng.random() < self.config.crossover_probability:
                    parent_b = self._tournament(population, self.config.tournament_size, rng)
                    tree, _ = crossover_trees(
                        tree,
                        parent_b.tree,
                        rng,
                        max_height=self.config.max_tree_height,
                    )
                if rng.random() < self.config.mutation_probability:
                    tree = mutate_tree(
                        tree,
                        self.pset,
                        rng,
                        max_depth=self.config.max_depth,
                        max_height=self.config.max_tree_height,
                    )
                candidate = evaluate(tree, generation)
                if candidate is None:
                    stop_reason = "normalized-work-budget-exhausted"
                    break
                offspring.append(candidate)
            # Standard generational elitism: retain each elite exactly once and fill
            # the remaining slots only with newly evaluated offspring.  In
            # particular, elites are neither re-evaluated nor duplicated in a
            # population through a parent+children merge.
            population = [*elites, *offspring]
            population_history.append(tuple(candidate.proposal_seq for candidate in population))
            if stop_reason == "normalized-work-budget-exhausted":
                break

        feasible = [
            candidate
            for candidate in candidates
            if candidate.result.promoted_to_l2
            and candidate.result.evaluation is not None
            and candidate.result.evaluation.fitness.feasible
        ]
        champion = (
            min(feasible, key=lambda candidate: candidate.selection_key) if feasible else None
        )
        return GPHHRun(
            root_seed=root_seed,
            config=self.config,
            candidates=tuple(candidates),
            champion=champion,
            spent_microwork=spent,
            stop_reason=stop_reason,
            population_proposal_seqs_by_generation=tuple(population_history),
        )
