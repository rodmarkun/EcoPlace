"""Contextual per-instance assignment oracles for TODO 7.7.

These methods return assignment vectors, not reusable scheduling policies, and are
therefore never wrapped in ``PolicyBundleAdapter``.  FFD is the actual queue-ordered
heuristic. NSGA-II supplies exploratory offline Pareto context. HiGHS solves a
scalarized static-proxy MILP exactly on explicitly small instances and rejects every
non-optimal solver termination. None of these proxy objectives is a C20 lower bound.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from deap import base, tools


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class AssignmentInstance:
    instance_id: str
    nodes: tuple[dict[str, Any], ...]
    pods: tuple[dict[str, Any], ...]
    graph: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.instance_id:
            raise ValueError("instance_id must be non-empty")
        encoded = _canonical(
            {
                "nodes": self.nodes,
                "pods": self.pods,
                "graph": self.graph,
            }
        )
        payload = json.loads(encoded)
        nodes = tuple(payload["nodes"])
        pods = tuple(payload["pods"])
        graph = payload["graph"]
        if not nodes or not pods:
            raise ValueError("assignment instance requires nodes and pods")
        if [node.get("id") for node in nodes] != list(range(len(nodes))):
            raise ValueError("assignment node ids must be dense and ordered")
        if [pod.get("id") for pod in pods] != list(range(len(pods))):
            raise ValueError("assignment pod ids must be dense and ordered")
        for node in nodes:
            if (
                node["cpu_allocated"] > node["cpu_capacity"]
                or node["mem_allocated"] > node["mem_capacity"]
            ):
                raise ValueError("fixed background allocation exceeds node capacity")
            if len(node["latency_to_ms"]) != len(nodes):
                raise ValueError("each node needs one latency per assignment node")
            for name in ("power_idle_w", "power_max_w"):
                if not math.isfinite(node[name]) or node[name] < 0:
                    raise ValueError(f"node {name} must be finite and non-negative")
            if node["power_max_w"] < node["power_idle_w"]:
                raise ValueError("node maximum power must be >= idle power")
        if set(graph) != {"src", "dst", "rate", "payload_kb"}:
            raise ValueError("assignment graph has an invalid shape")
        edge_count = len(graph["src"])
        if any(len(graph[key]) != edge_count for key in ("dst", "rate", "payload_kb")):
            raise ValueError("assignment graph columns must have equal length")
        if any(not 0 <= endpoint < len(pods) for endpoint in (*graph["src"], *graph["dst"])):
            raise ValueError("assignment graph endpoint is outside the pod set")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "pods", pods)
        object.__setattr__(self, "graph", graph)
        # Fail early when a pod cannot fit even on an otherwise empty eligible node.
        for pod in pods:
            if not any(
                not node["suspended"]
                and node["cpu_allocated"] + pod["cpu_request"] <= node["cpu_capacity"]
                and node["mem_allocated"] + pod["mem_request"] <= node["mem_capacity"]
                for node in nodes
            ):
                raise ValueError(f"pod {pod['id']} cannot fit on any eligible node")

    @property
    def sha256(self) -> str:
        return _sha(
            {
                "instance_id": self.instance_id,
                "nodes": self.nodes,
                "pods": self.pods,
                "graph": self.graph,
                "allocation_scope": (
                    "pods are decision variables; node allocations are fixed background"
                ),
            }
        )


def assignment_objectives(
    instance: AssignmentInstance, assignment: tuple[int, ...]
) -> tuple[float, float]:
    if len(assignment) != len(instance.pods) or any(
        not isinstance(node_id, int)
        or isinstance(node_id, bool)
        or not 0 <= node_id < len(instance.nodes)
        for node_id in assignment
    ):
        raise ValueError("assignment must map every pod to one valid node id")
    cpu = [node["cpu_allocated"] for node in instance.nodes]
    memory = [node["mem_allocated"] for node in instance.nodes]
    for pod, node_id in zip(instance.pods, assignment, strict=True):
        node = instance.nodes[node_id]
        if node["suspended"]:
            raise ValueError("assignment places a pod on a suspended node")
        cpu[node_id] += pod["cpu_request"]
        memory[node_id] += pod["mem_request"]
    if any(
        cpu[index] > node["cpu_capacity"] or memory[index] > node["mem_capacity"]
        for index, node in enumerate(instance.nodes)
    ):
        raise ValueError("assignment exceeds node capacity")
    energy_proxy = 0.0
    for index, node in enumerate(instance.nodes):
        if cpu[index] == 0 and memory[index] == 0:
            continue
        span = node["power_max_w"] - node["power_idle_w"]
        energy_proxy += node["power_idle_w"] + span * (cpu[index] / max(node["cpu_capacity"], 1))
    network_cost = math.fsum(
        rate * payload * instance.nodes[assignment[src]]["latency_to_ms"][assignment[dst]]
        for src, dst, rate, payload in zip(
            instance.graph["src"],
            instance.graph["dst"],
            instance.graph["rate"],
            instance.graph["payload_kb"],
            strict=True,
        )
    )
    return energy_proxy, network_cost


@dataclass(frozen=True)
class FFDAssignment:
    instance_sha256: str
    assignment: tuple[int, ...]
    decreasing_order: tuple[int, ...]
    energy_proxy_w: float
    network_cost_kb_ms_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-ffd-assignment-v1",
            "role": "contextual-full-queue-heuristic-not-reusable-policy",
            "instance_sha256": self.instance_sha256,
            "decreasing_key": "max(cpu_fraction, memory_fraction), pod-id",
            "decreasing_order": list(self.decreasing_order),
            "assignment": list(self.assignment),
            "energy_proxy_w": self.energy_proxy_w,
            "network_cost_kb_ms_s": self.network_cost_kb_ms_s,
            "c20_lower_bound": False,
        }
        return {**payload, "sha256": _sha(payload)}


class FirstFitDecreasingAssignmentOracle:
    """Sort the complete pod queue by size, then first-fit in stable node-id order."""

    def run(self, instance: AssignmentInstance) -> FFDAssignment:
        cpu = [node["cpu_allocated"] for node in instance.nodes]
        memory = [node["mem_allocated"] for node in instance.nodes]
        assignment = [-1] * len(instance.pods)
        order = tuple(
            sorted(
                range(len(instance.pods)),
                key=lambda pod_id: (
                    max(
                        instance.pods[pod_id]["cpu_request"]
                        / max(
                            max(node["cpu_capacity"] for node in instance.nodes),
                            1,
                        ),
                        instance.pods[pod_id]["mem_request"]
                        / max(
                            max(node["mem_capacity"] for node in instance.nodes),
                            1,
                        ),
                    ),
                    -pod_id,
                ),
                reverse=True,
            )
        )
        for pod_id in order:
            pod = instance.pods[pod_id]
            selected = next(
                (
                    node["id"]
                    for node in instance.nodes
                    if not node["suspended"]
                    and cpu[node["id"]] + pod["cpu_request"] <= node["cpu_capacity"]
                    and memory[node["id"]] + pod["mem_request"] <= node["mem_capacity"]
                ),
                None,
            )
            if selected is None:
                raise ValueError("FFD could not construct a feasible assignment")
            assignment[pod_id] = selected
            cpu[selected] += pod["cpu_request"]
            memory[selected] += pod["mem_request"]
        energy, network = assignment_objectives(instance, tuple(assignment))
        return FFDAssignment(
            instance_sha256=instance.sha256,
            assignment=tuple(assignment),
            decreasing_order=order,
            energy_proxy_w=energy,
            network_cost_kb_ms_s=network,
        )


class _FitnessMin2(base.Fitness):
    weights = (-1.0, -1.0)


class _AssignmentIndividual(list[int]):
    def __init__(self, values: tuple[int, ...] | list[int]) -> None:
        super().__init__(values)
        self.fitness = _FitnessMin2()


@dataclass(frozen=True)
class NSGA2Config:
    population_size: int = 40
    generations: int = 40
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    max_initialization_attempts: int = 100

    def __post_init__(self) -> None:
        if (
            not isinstance(self.population_size, int)
            or isinstance(self.population_size, bool)
            or self.population_size < 4
            or self.population_size % 4 != 0
        ):
            raise ValueError("NSGA-II population_size must be a positive multiple of 4")
        for name in ("generations", "max_initialization_attempts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    @property
    def objective_evaluation_budget(self) -> int:
        return self.population_size * (self.generations + 1)


@dataclass(frozen=True)
class AssignmentSolution:
    assignment: tuple[int, ...]
    energy_proxy_w: float
    network_cost_kb_ms_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment": list(self.assignment),
            "energy_proxy_w": self.energy_proxy_w,
            "network_cost_kb_ms_s": self.network_cost_kb_ms_s,
        }


@dataclass(frozen=True)
class NSGA2Run:
    root_seed: int
    instance_sha256: str
    config: NSGA2Config
    objective_evaluations: int
    pareto: tuple[AssignmentSolution, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-nsga2-assignment-v1",
            "role": "contextual-offline-instance-assignment-not-reusable-policy",
            "root_seed": self.root_seed,
            "instance_sha256": self.instance_sha256,
            "config": dataclasses.asdict(self.config),
            "objective_evaluation_budget": self.config.objective_evaluation_budget,
            "objective_evaluations": self.objective_evaluations,
            "objectives": ["energy_proxy_w", "network_cost_kb_ms_s"],
            "objective_scope": "static assignment proxy",
            "c20_lower_bound": False,
            "pareto": [solution.to_dict() for solution in self.pareto],
        }
        return {**payload, "sha256": _sha(payload)}


class NSGA2AssignmentOracle:
    def __init__(self, config: NSGA2Config | None = None) -> None:
        self.config = config or NSGA2Config()

    @staticmethod
    def _random_feasible(
        instance: AssignmentInstance, rng: random.Random
    ) -> tuple[int, ...] | None:
        cpu = [node["cpu_allocated"] for node in instance.nodes]
        memory = [node["mem_allocated"] for node in instance.nodes]
        assignment = [-1] * len(instance.pods)
        order = list(range(len(instance.pods)))
        rng.shuffle(order)
        order.sort(
            key=lambda pod_id: max(
                instance.pods[pod_id]["cpu_request"],
                instance.pods[pod_id]["mem_request"],
            ),
            reverse=True,
        )
        for pod_id in order:
            pod = instance.pods[pod_id]
            feasible = [
                node["id"]
                for node in instance.nodes
                if not node["suspended"]
                and cpu[node["id"]] + pod["cpu_request"] <= node["cpu_capacity"]
                and memory[node["id"]] + pod["mem_request"] <= node["mem_capacity"]
            ]
            if not feasible:
                return None
            node_id = rng.choice(feasible)
            assignment[pod_id] = node_id
            cpu[node_id] += pod["cpu_request"]
            memory[node_id] += pod["mem_request"]
        return tuple(assignment)

    def _repair(
        self,
        instance: AssignmentInstance,
        proposed: list[int],
        rng: random.Random,
    ) -> tuple[int, ...] | None:
        cpu = [node["cpu_allocated"] for node in instance.nodes]
        memory = [node["mem_allocated"] for node in instance.nodes]
        repaired = [-1] * len(instance.pods)
        order = sorted(
            range(len(instance.pods)),
            key=lambda pod_id: (
                max(
                    instance.pods[pod_id]["cpu_request"],
                    instance.pods[pod_id]["mem_request"],
                ),
                -pod_id,
            ),
            reverse=True,
        )
        for pod_id in order:
            pod = instance.pods[pod_id]
            feasible = [
                node["id"]
                for node in instance.nodes
                if not node["suspended"]
                and cpu[node["id"]] + pod["cpu_request"] <= node["cpu_capacity"]
                and memory[node["id"]] + pod["mem_request"] <= node["mem_capacity"]
            ]
            if not feasible:
                return None
            preferred = proposed[pod_id]
            node_id = preferred if preferred in feasible else rng.choice(feasible)
            repaired[pod_id] = node_id
            cpu[node_id] += pod["cpu_request"]
            memory[node_id] += pod["mem_request"]
        return tuple(repaired)

    @staticmethod
    def _tournament(
        population: list[_AssignmentIndividual], rng: random.Random
    ) -> _AssignmentIndividual:
        first, second = rng.sample(population, 2)
        if first.fitness.dominates(second.fitness):
            return first
        if second.fitness.dominates(first.fitness):
            return second
        first_crowding = getattr(first.fitness, "crowding_dist", 0.0)
        second_crowding = getattr(second.fitness, "crowding_dist", 0.0)
        if first_crowding != second_crowding:
            return first if first_crowding > second_crowding else second
        return first if rng.random() < 0.5 else second

    def run(self, instance: AssignmentInstance, *, root_seed: int) -> NSGA2Run:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        rng = random.Random(root_seed)
        population = []
        attempts = 0
        while len(population) < self.config.population_size:
            attempts += 1
            if attempts > self.config.max_initialization_attempts:
                raise ValueError("could not initialize a feasible NSGA-II population")
            assignment = self._random_feasible(instance, rng)
            if assignment is not None:
                population.append(_AssignmentIndividual(assignment))
        evaluations = 0

        def evaluate(individual: _AssignmentIndividual) -> None:
            nonlocal evaluations
            individual.fitness.values = assignment_objectives(instance, tuple(individual))
            evaluations += 1

        for individual in population:
            evaluate(individual)
        population = tools.selNSGA2(population, self.config.population_size)

        for _generation in range(self.config.generations):
            offspring = []
            while len(offspring) < self.config.population_size:
                parent_a = self._tournament(population, rng)
                parent_b = self._tournament(population, rng)
                left = list(parent_a)
                right = list(parent_b)
                if rng.random() < self.config.crossover_probability:
                    for index in range(len(left)):
                        if rng.random() < 0.5:
                            left[index], right[index] = right[index], left[index]
                for child, fallback in ((left, parent_a), (right, parent_b)):
                    if rng.random() < self.config.mutation_probability:
                        locus = rng.randrange(len(child))
                        child[locus] = rng.randrange(len(instance.nodes))
                    repaired = self._repair(instance, child, rng)
                    values = tuple(fallback) if repaired is None else repaired
                    individual = _AssignmentIndividual(values)
                    evaluate(individual)
                    offspring.append(individual)
                    if len(offspring) >= self.config.population_size:
                        break
            population = tools.selNSGA2([*population, *offspring], self.config.population_size)

        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        unique = {
            tuple(individual): AssignmentSolution(
                assignment=tuple(individual),
                energy_proxy_w=individual.fitness.values[0],
                network_cost_kb_ms_s=individual.fitness.values[1],
            )
            for individual in front
        }
        pareto = tuple(
            sorted(
                unique.values(),
                key=lambda solution: (
                    solution.energy_proxy_w,
                    solution.network_cost_kb_ms_s,
                    solution.assignment,
                ),
            )
        )
        if evaluations != self.config.objective_evaluation_budget:
            raise AssertionError("NSGA-II evaluation accounting drifted")
        return NSGA2Run(
            root_seed=root_seed,
            instance_sha256=instance.sha256,
            config=self.config,
            objective_evaluations=evaluations,
            pareto=pareto,
        )


@dataclass(frozen=True)
class MILPConfig:
    max_nodes: int = 8
    max_pods: int = 20
    max_edges: int = 40
    energy_weight: float = 1.0
    network_weight: float = 1.0
    time_limit_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("max_nodes", "max_pods", "max_edges"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        for name in ("energy_weight", "network_weight", "time_limit_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class MILPResult:
    root_seed: int
    instance_sha256: str
    config: MILPConfig
    highspy_version: str
    model_status: str
    assignment: AssignmentSolution
    scalar_objective: float
    variable_count: int
    constraint_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-highs-assignment-v1",
            "role": "contextual-small-instance-exact-scalarized-oracle",
            "root_seed": self.root_seed,
            "instance_sha256": self.instance_sha256,
            "config": dataclasses.asdict(self.config),
            "highspy_version": self.highspy_version,
            "model_status": self.model_status,
            "assignment": self.assignment.to_dict(),
            "scalar_objective": self.scalar_objective,
            "objective_scope": "exact optimum of static scalarized proxy only",
            "c20_lower_bound": False,
            "variable_count": self.variable_count,
            "constraint_count": self.constraint_count,
        }
        return {**payload, "sha256": _sha(payload)}


class HighsMILPOracle:
    def __init__(self, config: MILPConfig | None = None) -> None:
        self.config = config or MILPConfig()

    def solve(self, instance: AssignmentInstance, *, root_seed: int) -> MILPResult:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**31
        ):
            raise ValueError("HiGHS root_seed must be an explicit integer in [0, 2**31)")
        if (
            len(instance.nodes) > self.config.max_nodes
            or len(instance.pods) > self.config.max_pods
            or len(instance.graph["src"]) > self.config.max_edges
        ):
            raise ValueError("instance exceeds the registered small-MILP limits")
        import highspy

        model = highspy.Highs()
        for name, value in (
            ("output_flag", False),
            ("threads", 1),
            ("random_seed", root_seed),
            ("mip_rel_gap", 0.0),
            ("time_limit", self.config.time_limit_s),
        ):
            if model.setOptionValue(name, value) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS rejected option {name}")
        empty_i = np.asarray([], dtype=np.int32)
        empty_v = np.asarray([], dtype=np.float64)

        def binary(cost: float, *, lower: float = 0.0) -> int:
            index = model.getNumCol()
            status = model.addCol(cost, lower, 1.0, 0, empty_i, empty_v)
            if (
                status != highspy.HighsStatus.kOk
                or model.changeColIntegrality(index, highspy.HighsVarType.kInteger)
                != highspy.HighsStatus.kOk
            ):
                raise RuntimeError("HiGHS rejected a binary variable")
            return index

        def row(lower: float, upper: float, terms: dict[int, float]) -> None:
            indices = np.asarray(sorted(terms), dtype=np.int32)
            values = np.asarray([terms[index] for index in indices], dtype=np.float64)
            if model.addRow(lower, upper, len(indices), indices, values) != (
                highspy.HighsStatus.kOk
            ):
                raise RuntimeError("HiGHS rejected a model constraint")

        x = {}
        for pod in instance.pods:
            for node in instance.nodes:
                span = node["power_max_w"] - node["power_idle_w"]
                marginal = span * pod["cpu_request"] / max(node["cpu_capacity"], 1)
                upper = 0.0 if node["suspended"] else 1.0
                index = binary(self.config.energy_weight * marginal)
                if upper == 0.0:
                    row(0.0, 0.0, {index: 1.0})
                x[pod["id"], node["id"]] = index
        z = {}
        for node in instance.nodes:
            active_background = node["cpu_allocated"] > 0 or node["mem_allocated"] > 0
            z[node["id"]] = binary(
                self.config.energy_weight * node["power_idle_w"],
                lower=1.0 if active_background else 0.0,
            )
        y = {}
        for edge, (src, dst, rate, payload) in enumerate(
            zip(
                instance.graph["src"],
                instance.graph["dst"],
                instance.graph["rate"],
                instance.graph["payload_kb"],
                strict=True,
            )
        ):
            for source_node in instance.nodes:
                for target_node in instance.nodes:
                    network = rate * payload * source_node["latency_to_ms"][target_node["id"]]
                    index = binary(self.config.network_weight * network)
                    y[edge, source_node["id"], target_node["id"]] = index
                    source_x = x[src, source_node["id"]]
                    target_x = x[dst, target_node["id"]]
                    row(-highspy.kHighsInf, 0.0, {index: 1.0, source_x: -1.0})
                    row(-highspy.kHighsInf, 0.0, {index: 1.0, target_x: -1.0})
                    product_terms = {index: 1.0}
                    product_terms[source_x] = product_terms.get(source_x, 0.0) - 1.0
                    product_terms[target_x] = product_terms.get(target_x, 0.0) - 1.0
                    row(-1.0, highspy.kHighsInf, product_terms)
        for pod in instance.pods:
            row(
                1.0,
                1.0,
                {x[pod["id"], node["id"]]: 1.0 for node in instance.nodes},
            )
        for node in instance.nodes:
            node_id = node["id"]
            row(
                -highspy.kHighsInf,
                node["cpu_capacity"] - node["cpu_allocated"],
                {x[pod["id"], node_id]: float(pod["cpu_request"]) for pod in instance.pods},
            )
            row(
                -highspy.kHighsInf,
                node["mem_capacity"] - node["mem_allocated"],
                {x[pod["id"], node_id]: float(pod["mem_request"]) for pod in instance.pods},
            )
            activation = {x[pod["id"], node_id]: 1.0 for pod in instance.pods}
            activation[z[node_id]] = -float(len(instance.pods))
            row(-highspy.kHighsInf, 0.0, activation)
        if model.run() != highspy.HighsStatus.kOk:
            raise RuntimeError("HiGHS solve call failed")
        status = model.getModelStatus()
        if status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"HiGHS did not prove optimality: {status.name}")
        values = model.getSolution().col_value
        assignment = []
        for pod in instance.pods:
            selected = [
                node["id"] for node in instance.nodes if values[x[pod["id"], node["id"]]] > 0.5
            ]
            if len(selected) != 1:
                raise RuntimeError("HiGHS returned a non-integral assignment")
            assignment.append(selected[0])
        energy, network = assignment_objectives(instance, tuple(assignment))
        scalar = self.config.energy_weight * energy + self.config.network_weight * network
        if not math.isclose(scalar, model.getObjectiveValue(), rel_tol=1e-8, abs_tol=1e-8):
            raise RuntimeError("recomputed MILP objective differs from solver objective")
        solution = AssignmentSolution(tuple(assignment), energy, network)
        return MILPResult(
            root_seed=root_seed,
            instance_sha256=instance.sha256,
            config=self.config,
            highspy_version=model.version(),
            model_status=status.name,
            assignment=solution,
            scalar_objective=scalar,
            variable_count=model.getNumCol(),
            constraint_count=model.getNumRow(),
        )


__all__ = [
    "AssignmentInstance",
    "AssignmentSolution",
    "FFDAssignment",
    "FirstFitDecreasingAssignmentOracle",
    "HighsMILPOracle",
    "MILPConfig",
    "MILPResult",
    "NSGA2AssignmentOracle",
    "NSGA2Config",
    "NSGA2Run",
    "assignment_objectives",
]
