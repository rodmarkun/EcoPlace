"""Frozen graph-aware PPO baseline with sealed training provenance (TODO 7.4).

PPO is trained on action-conditioned placement episodes, not independent labelled
decisions. Each action changes node allocations and pod placement before the next
observation. Invalid actions are masked. Hyperparameter selection uses a disjoint
quarantined-pilot validation corpus, while the authoritative DES/C20 ladder remains
the fitness-of-record for the frozen reusable policy.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from analysis.tracking import derive_seed
from baselines.adapter import PolicyBundleAdapter
from sim.actuation import CallableProposer
from sim.scheduler import filter_feasible
from sim.snapshot import SnapshotLibrary

NODE_FEATURES = (
    "candidate",
    "cpu_used_after",
    "mem_used_after",
    "pod_cpu_fraction",
    "pod_mem_fraction",
    "idle_power_scaled",
    "marginal_power_scaled",
    "tier_scaled",
    "suspended",
    "neighbor_rate_latency_scaled",
    "neighbor_traffic_latency_scaled",
)
ARCHITECTURE_CONTRACT = "masked-ppo-fixed-node-graph-v1"
REWARD_CONTRACT = "negative-marginal-power-plus-traffic-latency-v1"
DECISION_STEP_UNIT = "placement-decision"
_HEX = re.compile(r"[0-9a-f]{64}")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_digest(label: str, value: str) -> None:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class GraphEncodingConfig:
    max_nodes: int
    latency_scale_ms: float = 1000.0
    traffic_scale_kb_ms_s: float = 1_000_000.0
    power_scale_w: float = 500.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_nodes, int)
            or isinstance(self.max_nodes, bool)
            or self.max_nodes < 1
        ):
            raise ValueError("max_nodes must be a positive int")
        for name in ("latency_scale_ms", "traffic_scale_kb_ms_s", "power_scale_w"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def observation_size(self) -> int:
        return self.max_nodes * len(NODE_FEATURES)


def _neighbor_costs(pod: dict, node: dict, ctx: dict) -> tuple[float, float]:
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
    return latency, traffic


class GraphObservationEncoder:
    def __init__(self, config: GraphEncodingConfig) -> None:
        self.config = config

    def encode(self, pod: dict, candidates: list[dict], ctx: dict) -> np.ndarray:
        if len(ctx["nodes"]) > self.config.max_nodes:
            raise ValueError("context exceeds encoder max_nodes")
        candidate_ids = {node["id"] for node in candidates}
        rows = np.zeros((self.config.max_nodes, len(NODE_FEATURES)), dtype=np.float32)
        for node in ctx["nodes"]:
            node_id = node["id"]
            if not 0 <= node_id < self.config.max_nodes:
                raise ValueError("node ids must be dense and within max_nodes")
            cpu_capacity = max(node["cpu_capacity"], 1)
            mem_capacity = max(node["mem_capacity"], 1)
            cpu_after = (node["cpu_allocated"] + pod["cpu_request"]) / cpu_capacity
            mem_after = (node["mem_allocated"] + pod["mem_request"]) / mem_capacity
            latency, traffic = _neighbor_costs(pod, node, ctx)
            power_span = max(node["power_max_w"] - node["power_idle_w"], 0.0)
            rows[node_id] = np.asarray(
                (
                    float(node_id in candidate_ids),
                    min(cpu_after, 2.0),
                    min(mem_after, 2.0),
                    min(pod["cpu_request"] / cpu_capacity, 2.0),
                    min(pod["mem_request"] / mem_capacity, 2.0),
                    min(node["power_idle_w"] / self.config.power_scale_w, 2.0),
                    min(
                        power_span * pod["cpu_request"] / cpu_capacity / self.config.power_scale_w,
                        2.0,
                    ),
                    min(float(node["tier"]) / 3.0, 2.0),
                    float(node["suspended"]),
                    min(latency / self.config.latency_scale_ms, 2.0),
                    min(traffic / self.config.traffic_scale_kb_ms_s, 2.0),
                ),
                dtype=np.float32,
            )
        return rows.reshape(-1)

    def candidate_mask(self, observation: np.ndarray) -> np.ndarray:
        rows = np.asarray(observation, dtype=np.float32).reshape(
            self.config.max_nodes, len(NODE_FEATURES)
        )
        return rows[:, 0] > 0.5


@dataclass(frozen=True)
class PlacementEpisode:
    """One sealed, ordered placement trajectory source."""

    episode_id: str
    source_library_sha256: str
    source_episode_sha256: str
    data_role: str
    split_role: str
    decision_start_step: int
    ctx: dict[str, Any]
    pending_pod_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id is required")
        _require_digest("source_library_sha256", self.source_library_sha256)
        _require_digest("source_episode_sha256", self.source_episode_sha256)
        if self.data_role not in {"quarantined-pilot", "online-history-before-step"}:
            raise ValueError("invalid PPO episode data role")
        if self.split_role not in {"train", "validation"}:
            raise ValueError("invalid PPO corpus split role")
        if (
            not isinstance(self.decision_start_step, int)
            or isinstance(self.decision_start_step, bool)
            or self.decision_start_step < 0
        ):
            raise ValueError("decision_start_step must be a non-negative int")
        payload = json.loads(_canonical(self.ctx))
        pending = tuple(self.pending_pod_ids)
        if not pending or len(pending) != len(set(pending)):
            raise ValueError("episode needs unique pending pod ids")
        pods = payload.get("pods", [])
        if any(
            not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(pods)
            for index in pending
        ):
            raise ValueError("pending pod id is outside episode context")
        if any(pods[index]["current_node"] >= 0 for index in pending):
            raise ValueError("pending PPO pods must initially be unplaced")
        object.__setattr__(self, "ctx", payload)
        object.__setattr__(self, "pending_pod_ids", pending)

    @property
    def decision_end_step(self) -> int:
        return self.decision_start_step + len(self.pending_pod_ids) - 1

    @property
    def sha256(self) -> str:
        return _sha(dataclasses.asdict(self))

    @property
    def trajectory_sha256(self) -> str:
        """Content identity independent of caller-supplied provenance labels."""
        return _sha(
            {
                "ctx": self.ctx,
                "pending_pod_ids": self.pending_pod_ids,
            }
        )


@dataclass(frozen=True)
class PPOCorpus:
    data_role: str
    split_role: str
    episodes: tuple[PlacementEpisode, ...]

    def __post_init__(self) -> None:
        if self.data_role not in {"quarantined-pilot", "online-history-before-step"}:
            raise ValueError("invalid PPO corpus data role")
        if self.split_role not in {"train", "validation"}:
            raise ValueError("invalid PPO corpus split role")
        if not self.episodes:
            raise ValueError("PPO corpus needs at least one episode")
        if any(
            episode.data_role != self.data_role or episode.split_role != self.split_role
            for episode in self.episodes
        ):
            raise ValueError("episode provenance does not match corpus role/split")
        ids = [episode.episode_id for episode in self.episodes]
        if len(ids) != len(set(ids)):
            raise ValueError("PPO corpus episode ids must be unique")

    @property
    def sha256(self) -> str:
        return _sha([episode.sha256 for episode in self.episodes])

    @property
    def source_library_sha256(self) -> tuple[str, ...]:
        return tuple(sorted({episode.source_library_sha256 for episode in self.episodes}))

    @property
    def decision_count(self) -> int:
        return sum(len(episode.pending_pod_ids) for episode in self.episodes)

    @property
    def trajectory_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted({episode.trajectory_sha256 for episode in self.episodes}))


def corpus_from_snapshot_library(
    library: SnapshotLibrary,
    *,
    split_role: str,
) -> PPOCorpus:
    data_role = library.meta.get("data_role")
    if data_role not in {"quarantined-pilot", "online-history-before-step"}:
        raise ValueError("PPO snapshot library has an invalid sealed data_role")
    episode_digest = library.meta.get("episode_sha256")
    _require_digest("snapshot-library episode_sha256", episode_digest)
    if library.meta.get("decision_step_unit") != DECISION_STEP_UNIT:
        raise ValueError(
            f"PPO snapshot library must declare decision_step_unit={DECISION_STEP_UNIT!r}"
        )
    ranges: list[tuple[int, int]] | None = None
    if data_role == "online-history-before-step":
        raw_ranges = library.meta.get("decision_step_ranges")
        if not isinstance(raw_ranges, list) or len(raw_ranges) != len(library.snapshots):
            raise ValueError(
                "online PPO history needs one sealed decision_step_range per snapshot"
            )
        ranges = []
        previous_start = -1
        previous_end = 0
        for index, raw_range in enumerate(raw_ranges):
            if not isinstance(raw_range, dict) or set(raw_range) != {
                "start",
                "end_exclusive",
            }:
                raise ValueError("decision_step_ranges need exact start/end_exclusive fields")
            start = raw_range["start"]
            end = raw_range["end_exclusive"]
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end < start
                or start <= previous_start
                or (index > 0 and start < previous_end)
            ):
                raise ValueError(
                    "decision_step_ranges must be non-negative, ordered, and non-overlapping"
                )
            pending = len(library.snapshots[index].pending_dense_ids)
            if end - start != pending:
                raise ValueError("sealed decision-step range width must match pending decisions")
            ranges.append((start, end))
            previous_start = start
            previous_end = end
        decision_cursor = 0
    else:
        decision_cursor = library.meta.get("decision_start_step", 0)
        if (
            not isinstance(decision_cursor, int)
            or isinstance(decision_cursor, bool)
            or decision_cursor < 0
        ):
            raise ValueError("snapshot-library decision_start_step must be non-negative")
    library_digest = library.digest()
    episodes = []
    for snapshot_index, snapshot in enumerate(library.snapshots):
        if ranges is not None:
            decision_cursor = ranges[snapshot_index][0]
        if not snapshot.pending_dense_ids:
            continue
        plain = snapshot.to_dict()["ctx"]
        episodes.append(
            PlacementEpisode(
                episode_id=f"{episode_digest}:snapshot:{snapshot_index}",
                source_library_sha256=library_digest,
                source_episode_sha256=episode_digest,
                data_role=data_role,
                split_role=split_role,
                decision_start_step=decision_cursor,
                ctx=plain,
                pending_pod_ids=tuple(snapshot.pending_dense_ids),
            )
        )
        if ranges is None:
            decision_cursor += len(snapshot.pending_dense_ids)
    return PPOCorpus(data_role, split_role, tuple(episodes))


class ActionLogitModel(Protocol):
    artifact_sha256: str

    def action_logits(self, observation: np.ndarray) -> np.ndarray: ...


class StableBaselinesPPOModel:
    """Read-only wrapper around a frozen MaskablePPO artifact."""

    def __init__(self, artifact: str | Path) -> None:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        from sb3_contrib import MaskablePPO

        self.path = path
        self.artifact_sha256 = _sha256_file(path)
        self._model = MaskablePPO.load(path, device="cpu")
        self._model.policy.set_training_mode(False)

    def action_logits(self, observation: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor, _ = self._model.policy.obs_to_tensor(observation)
            distribution = self._model.policy.get_distribution(tensor)
            logits = distribution.distribution.logits.detach().cpu().numpy()[0]
        return np.asarray(logits, dtype=np.float64)


class FrozenPPOScorer:
    def __init__(self, model: ActionLogitModel, encoder: GraphObservationEncoder) -> None:
        self.model = model
        self.encoder = encoder
        self.name = f"ppo:{model.artifact_sha256[:12]}"

    def score_batch(self, pod: dict, nodes: list[dict], ctx: dict) -> list[float]:
        observation = self.encoder.encode(pod, nodes, ctx)
        logits = np.asarray(self.model.action_logits(observation), dtype=np.float64)
        expected = self.encoder.config.max_nodes + 1
        if logits.shape != (expected,) or not np.isfinite(logits).all():
            raise ValueError("frozen PPO returned malformed action logits")
        return [float(logits[node["id"]]) for node in nodes]


class FrozenPPOProposer:
    def __init__(self, scorer: FrozenPPOScorer) -> None:
        self.scorer = scorer

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
            # One counterfactual state: remove the current pod once, derive one
            # candidate mask, and compare every action from one model invocation.
            by_id[current]["cpu_allocated"] -= pod["cpu_request"]
            by_id[current]["mem_allocated"] -= pod["mem_request"]
            feasible_ids = filter_feasible(pod, nodes)
            candidates = [by_id[node_id] for node_id in feasible_ids]
            if current not in feasible_ids:
                by_id[current]["cpu_allocated"] += pod["cpu_request"]
                by_id[current]["mem_allocated"] += pod["mem_request"]
                continue
            scores = self.scorer.score_batch(pod, candidates, ctx)
            best_index = max(
                range(len(candidates)), key=lambda index: (scores[index], -candidates[index]["id"])
            )
            best = candidates[best_index]
            current_index = feasible_ids.index(current)
            if best["id"] != current and scores[best_index] > scores[current_index]:
                out.append({"kind": 0, "pod": pod["id"], "dest": best["id"]})
                pod["current_node"] = best["id"]
                best["cpu_allocated"] += pod["cpu_request"]
                best["mem_allocated"] += pod["mem_request"]
            else:
                by_id[current]["cpu_allocated"] += pod["cpu_request"]
                by_id[current]["mem_allocated"] += pod["mem_request"]
        return out


def frozen_ppo_bundle(
    model: ActionLogitModel,
    encoding: GraphEncodingConfig,
    *,
    training_receipt_sha256: str,
    registration_status: str,
) -> PolicyBundleAdapter:
    _require_digest("training_receipt_sha256", training_receipt_sha256)
    _require_digest("model artifact_sha256", model.artifact_sha256)
    if registration_status not in {"development-provisional", "p2-frozen", "exploratory-only"}:
        raise ValueError("invalid PPO bundle registration status")
    encoder = GraphObservationEncoder(encoding)
    scorer = FrozenPPOScorer(model, encoder)
    return PolicyBundleAdapter(
        name="ppo_frozen",
        version=f"artifact-{model.artifact_sha256[:12]}",
        scorer=scorer,
        proposer=CallableProposer("ppo_frozen", FrozenPPOProposer(scorer).propose_moves),
        representation="sb3-contrib-maskable-ppo-frozen-weights",
        artifact_sha256=model.artifact_sha256,
        metadata={
            "status": registration_status,
            "architecture_contract": ARCHITECTURE_CONTRACT,
            "architecture_qualification": (
                "mu-DDRL-inspired fixed graph features; not a reproduction"
            ),
            "reward_contract": REWARD_CONTRACT,
            "artifact_sha256": model.artifact_sha256,
            "training_receipt_sha256": training_receipt_sha256,
            "encoding": dataclasses.asdict(encoding),
            "node_features": list(NODE_FEATURES),
            "action_masking": "sb3-contrib-MaskablePPO",
            "inference_device": "cpu",
        },
    )


@dataclass(frozen=True)
class PPOHyperparameters:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    hidden_sizes: tuple[int, ...]
    total_timesteps: int

    def __post_init__(self) -> None:
        for name in ("learning_rate", "gamma", "ent_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0 or not 0 < self.gamma <= 1:
            raise ValueError("learning_rate must be positive and gamma within (0, 1]")
        for name in ("n_steps", "batch_size", "n_epochs", "total_timesteps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if self.n_steps % self.batch_size != 0:
            raise ValueError("n_steps must be divisible by batch_size")
        if not self.hidden_sizes or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive ints")


@dataclass(frozen=True)
class PPOTuningBudget:
    registration_status: str
    registration_source: str
    data_role: str
    trials: tuple[PPOHyperparameters, ...]
    max_total_timesteps: int

    def __post_init__(self) -> None:
        if self.registration_status not in {"development-provisional", "p2-frozen"}:
            raise ValueError("unknown PPO tuning registration status")
        if not self.registration_source:
            raise ValueError("PPO tuning registration_source is required")
        if self.data_role not in {"quarantined-pilot", "online-history-before-step"}:
            raise ValueError("unknown PPO training data role")
        if not self.trials:
            raise ValueError("PPO tuning budget needs at least one explicit trial")
        if self.data_role == "online-history-before-step" and len(self.trials) != 1:
            raise ValueError("online PPO retraining requires one P2-fixed hyperparameter trial")
        if (
            not isinstance(self.max_total_timesteps, int)
            or isinstance(self.max_total_timesteps, bool)
            or self.max_total_timesteps < 1
        ):
            raise ValueError("max_total_timesteps must be a positive int")
        if sum(trial.total_timesteps for trial in self.trials) > self.max_total_timesteps:
            raise ValueError("PPO trial grid exceeds max_total_timesteps")

    @property
    def confirmatory_ready(self) -> bool:
        return self.registration_status == "p2-frozen"


def load_tuning_budget(path: str | Path) -> tuple[PPOTuningBudget, GraphEncodingConfig]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != "evolved-continuum-ppo-tuning-v1":
        raise ValueError("unknown PPO tuning schema")
    if (
        raw.get("architecture_contract") != ARCHITECTURE_CONTRACT
        or raw.get("reward_contract") != REWARD_CONTRACT
    ):
        raise ValueError("PPO tuning file changed the model/reward contract")
    budget = PPOTuningBudget(
        registration_status=raw["registration_status"],
        registration_source=raw["registration_source"],
        data_role=raw["data_role"],
        trials=tuple(
            PPOHyperparameters(**{**trial, "hidden_sizes": tuple(trial["hidden_sizes"])})
            for trial in raw["trials"]
        ),
        max_total_timesteps=raw["max_total_timesteps"],
    )
    return budget, GraphEncodingConfig(**raw["encoding"])


def _action_reward(encoder: GraphObservationEncoder, observation: np.ndarray, action: int) -> float:
    rows = observation.reshape(encoder.config.max_nodes, len(NODE_FEATURES))
    if action == encoder.config.max_nodes:
        return -25.0
    if not 0 <= action < encoder.config.max_nodes or rows[action, 0] < 0.5:
        return -100.0
    row = rows[action]
    return -float(row[6] + row[10] + 0.25 * row[9])


def _make_training_env(corpus: PPOCorpus, encoder: GraphObservationEncoder, *, seed: int):
    import gymnasium as gym

    class PlacementEpisodeEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=2.0, shape=(encoder.config.observation_size,), dtype=np.float32
            )
            self.action_space = gym.spaces.Discrete(encoder.config.max_nodes + 1)
            self.order = np.arange(len(corpus.episodes))
            self.cursor = -1
            self.ctx: dict[str, Any] = {}
            self.pending: tuple[int, ...] = ()
            self.index = 0
            self._last_candidates: list[dict] = []

        def _state(self) -> tuple[dict, list[dict], np.ndarray]:
            pod = self.ctx["pods"][self.pending[self.index]]
            feasible = filter_feasible(pod, self.ctx["nodes"])
            by_id = {node["id"]: node for node in self.ctx["nodes"]}
            self._last_candidates = [by_id[node_id] for node_id in feasible]
            return pod, self._last_candidates, encoder.encode(pod, self._last_candidates, self.ctx)

        def action_masks(self) -> np.ndarray:
            _pod, _candidates, observation = self._state()
            node_mask = encoder.candidate_mask(observation)
            return np.concatenate((node_mask, np.asarray([not node_mask.any()])))

        def reset(self, *, seed: int, options: dict | None = None):
            if (
                not isinstance(seed, int)
                or isinstance(seed, bool)
                or not 0 <= seed < 2**63
            ):
                raise ValueError("PPO environment reset seed must be an explicit int in [0, 2**63)")
            continue_order = False
            if options is not None:
                if set(options) != {"_continue_registered_order"}:
                    raise ValueError("unknown PPO environment reset options")
                continue_order = options["_continue_registered_order"]
                if not isinstance(continue_order, bool):
                    raise ValueError("_continue_registered_order must be a bool")
            super().reset(seed=seed)
            if continue_order:
                if self.cursor < 0:
                    raise ValueError("cannot continue PPO episode order before a seeded reset")
            else:
                self.order = self.np_random.permutation(len(corpus.episodes))
                self.cursor = -1
            self.cursor = (self.cursor + 1) % len(self.order)
            episode = corpus.episodes[int(self.order[self.cursor])]
            self.ctx = copy.deepcopy(episode.ctx)
            self.pending = episode.pending_pod_ids
            self.index = 0
            return self._state()[2], {}

        def step(self, action: int):
            pod, candidates, observation = self._state()
            masks = self.action_masks()
            action = int(action)
            if not 0 <= action < len(masks) or not masks[action]:
                raise ValueError("MaskablePPO selected an invalid placement action")
            reward = _action_reward(encoder, observation, action)
            if action < encoder.config.max_nodes:
                selected = next(node for node in candidates if node["id"] == action)
                selected["cpu_allocated"] += pod["cpu_request"]
                selected["mem_allocated"] += pod["mem_request"]
                pod["current_node"] = action
            self.index += 1
            terminated = self.index >= len(self.pending)
            next_observation = (
                np.zeros(self.observation_space.shape, dtype=np.float32)
                if terminated
                else self._state()[2]
            )
            return next_observation, reward, terminated, False, {"decision_index": self.index - 1}

    # Validate the root at construction even though the first stochastic action is
    # the caller's required reset. This prevents a malformed seed from surviving
    # until an optional training dependency invokes the environment.
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("PPO environment seed must be an explicit int in [0, 2**63)")
    return PlacementEpisodeEnv()


def _make_sb3_training_env(
    corpus: PPOCorpus,
    encoder: GraphObservationEncoder,
    *,
    seed: int,
):
    """Adapt SB3's no-argument auto-reset protocol to the strict seeded API.

    Stable-Baselines3 calls ``env.reset()`` after terminal observations. The
    experiment-facing environment above deliberately rejects that call. This
    private adapter owns an explicitly seeded deterministic reset stream and
    supplies a derived seed on every SB3 reset, so the library protocol never
    reaches OS entropy and cannot weaken the public contract (R5-5).
    """
    import gymnasium as gym

    env = _make_training_env(corpus, encoder, seed=seed)

    class DeterministicallySeededAutoReset(gym.Wrapper):
        def __init__(self):
            super().__init__(env)
            self._root_seed = seed
            self._reset_index = 0

        def reset(self, **kwargs):
            requested_seed = kwargs.pop("seed", None)
            options = kwargs.pop("options", None)
            if kwargs:
                raise ValueError(f"unknown SB3 reset arguments: {sorted(kwargs)}")
            if requested_seed is not None and (
                not isinstance(requested_seed, int)
                or isinstance(requested_seed, bool)
                or not 0 <= requested_seed < 2**63
            ):
                raise ValueError("SB3 reset seed must be an int in [0, 2**63)")
            reset_seed = (
                requested_seed
                if requested_seed is not None
                else derive_seed(
                    self._root_seed,
                    "ppo",
                    "sb3-auto-reset",
                    self._reset_index,
                )
            )
            supplied_options = {} if options is None else dict(options)
            supplied_options["_continue_registered_order"] = self._reset_index > 0
            self._reset_index += 1
            return self.env.reset(seed=reset_seed, options=supplied_options)

    return DeterministicallySeededAutoReset()


def _validate_disjoint(train: PPOCorpus, validation: PPOCorpus, budget: PPOTuningBudget) -> None:
    if train.data_role != budget.data_role or validation.data_role != budget.data_role:
        raise ValueError("PPO corpora do not match the registered data role")
    if train.split_role != "train" or validation.split_role != "validation":
        raise ValueError("PPO requires explicit train and validation corpus roles")
    if set(train.source_library_sha256) & set(validation.source_library_sha256):
        raise ValueError("PPO train/validation snapshot libraries must be disjoint")
    train_sources = {episode.source_episode_sha256 for episode in train.episodes}
    validation_sources = {episode.source_episode_sha256 for episode in validation.episodes}
    if train_sources & validation_sources:
        raise ValueError("PPO train/validation source episodes must be disjoint")
    if set(train.trajectory_sha256s) & set(validation.trajectory_sha256s):
        raise ValueError("PPO train/validation trajectory content must be disjoint")


def _evaluate_model(
    model: Any,
    corpus: PPOCorpus,
    encoder: GraphObservationEncoder,
    *,
    seed: int,
) -> float:
    env = _make_training_env(corpus, encoder, seed=seed)
    rewards = []
    for episode_index in range(len(corpus.episodes)):
        episode_seed = derive_seed(seed, "ppo", "validation-reset", episode_index)
        observation, _ = env.reset(
            seed=episode_seed,
            options={"_continue_registered_order": episode_index > 0},
        )
        terminated = False
        while not terminated:
            action, _state = model.predict(
                observation, deterministic=True, action_masks=env.action_masks()
            )
            observation, reward, terminated, _truncated, _info = env.step(int(action))
            rewards.append(reward)
    env.close()
    return math.fsum(rewards) / len(rewards)


def tune_and_train_frozen_ppo(
    training: PPOCorpus,
    validation: PPOCorpus,
    budget: PPOTuningBudget,
    encoding: GraphEncodingConfig,
    output_dir: str | Path,
    *,
    root_seed: int,
    resume: bool = False,
) -> dict[str, Any]:
    """Train on sealed episodes and select only on a disjoint pilot validation set.

    When ``resume`` is true, each completed trial is durably checkpointed and verified
    before it is reused.  An interrupted in-flight trial is deliberately restarted from
    its explicit seed; completed trials are never silently retrained or overwritten.
    """
    if not isinstance(root_seed, int) or isinstance(root_seed, bool) or not 0 <= root_seed < 2**63:
        raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
    _validate_disjoint(training, validation, budget)
    if any(
        len(episode.ctx["nodes"]) > encoding.max_nodes
        for corpus in (training, validation)
        for episode in corpus.episodes
    ):
        raise ValueError("PPO episode exceeds encoding.max_nodes")
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    encoder = GraphObservationEncoder(encoding)
    checkpoint_path = output / "training_checkpoint.json"
    receipt_path = output / "training_receipt.json"
    budget_record = json.loads(_canonical(dataclasses.asdict(budget)))
    encoding_record = json.loads(_canonical(dataclasses.asdict(encoding)))
    checkpoint_contract = {
        "schema": "evolved-continuum-ppo-training-checkpoint-v1",
        "training_corpus_sha256": training.sha256,
        "validation_corpus_sha256": validation.sha256,
        "budget": budget_record,
        "encoding": encoding_record,
        "root_seed": root_seed,
    }
    checkpoint_contract_sha256 = _sha(checkpoint_contract)

    def write_checkpoint(trials: list[dict[str, Any]], *, status: str) -> None:
        payload = {
            **checkpoint_contract,
            "contract_sha256": checkpoint_contract_sha256,
            "status": status,
            "completed_trials": trials,
        }
        checkpoint = {**payload, "sha256": _sha(payload)}
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(_canonical(checkpoint) + "\n")
        os.replace(temporary, checkpoint_path)

    trials: list[dict[str, Any]] = []
    if receipt_path.exists():
        if not resume:
            raise FileExistsError(f"PPO training receipt already exists: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        load_frozen_ppo_bundle(
            output,
            expected_receipt_sha256=receipt["sha256"],
            expected_data_role=budget.data_role,
            require_p2_frozen=budget.confirmatory_ready,
        )
        if (
            receipt["training_corpus_sha256"] != training.sha256
            or receipt["validation_corpus_sha256"] != validation.sha256
            or receipt["root_seed"] != root_seed
            or receipt["encoding"] != encoding_record
            or receipt["max_total_timesteps"] != budget.max_total_timesteps
            or [trial["hyperparameters"] for trial in receipt["trials"]]
            != [json.loads(_canonical(dataclasses.asdict(trial))) for trial in budget.trials]
        ):
            raise ValueError("existing PPO receipt differs from the requested resumable run")
        return receipt
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError(f"PPO training checkpoint already exists: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text())
        body = {key: value for key, value in checkpoint.items() if key != "sha256"}
        if checkpoint.get("sha256") != _sha(body):
            raise ValueError("PPO training checkpoint digest mismatch")
        contract = {
            key: checkpoint.get(key) for key in checkpoint_contract
        }
        if (
            contract != checkpoint_contract
            or checkpoint.get("contract_sha256") != checkpoint_contract_sha256
            or checkpoint.get("status") != "in-progress"
        ):
            raise ValueError("PPO training checkpoint differs from the requested run")
        raw_trials = checkpoint.get("completed_trials")
        if not isinstance(raw_trials, list):
            raise ValueError("PPO training checkpoint trials are malformed")
        for expected_index, trial in enumerate(raw_trials):
            if trial.get("trial_index") != expected_index:
                raise ValueError("PPO checkpoint trial sequence is not contiguous")
            artifact_name = trial.get("artifact")
            if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
                raise ValueError("PPO checkpoint artifact must be a safe basename")
            artifact = output / artifact_name
            if not artifact.is_file() or _sha256_file(artifact) != trial.get("artifact_sha256"):
                raise ValueError("PPO checkpoint artifact digest mismatch")
        trials = raw_trials
    for trial_index, hyperparameters in enumerate(budget.trials):
        if trial_index < len(trials):
            expected_hyperparameters = json.loads(
                _canonical(dataclasses.asdict(hyperparameters))
            )
            if trials[trial_index]["hyperparameters"] != expected_hyperparameters:
                raise ValueError("PPO checkpoint hyperparameters differ from the tuning budget")
            continue
        derived_seed = derive_seed(root_seed, "ppo", "tuning", trial_index)
        trial_seed = derived_seed % 2**32
        env = _make_sb3_training_env(training, encoder, seed=trial_seed)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=hyperparameters.learning_rate,
            n_steps=hyperparameters.n_steps,
            batch_size=hyperparameters.batch_size,
            n_epochs=hyperparameters.n_epochs,
            gamma=hyperparameters.gamma,
            ent_coef=hyperparameters.ent_coef,
            policy_kwargs={"net_arch": list(hyperparameters.hidden_sizes)},
            seed=trial_seed,
            device="cpu",
            verbose=0,
        )
        model.learn(total_timesteps=hyperparameters.total_timesteps)
        artifact = output / f"trial-{trial_index:03d}.zip"
        partial_artifact = output / f"trial-{trial_index:03d}.partial.zip"
        model.save(partial_artifact)
        os.replace(partial_artifact, artifact)
        artifact_sha256 = _sha256_file(artifact)
        validation_reward = _evaluate_model(
            model,
            validation,
            encoder,
            seed=derive_seed(root_seed, "ppo", "validation", trial_index),
        )
        trials.append(
            {
                "trial_index": trial_index,
                "seed": trial_seed,
                "derived_seed": derived_seed,
                "hyperparameters": dataclasses.asdict(hyperparameters),
                "mean_validation_reward": validation_reward,
                "artifact": artifact.name,
                "artifact_sha256": artifact_sha256,
            }
        )
        env.close()
        write_checkpoint(trials, status="in-progress")
    champion = max(
        trials, key=lambda trial: (trial["mean_validation_reward"], -trial["trial_index"])
    )
    payload = {
        "schema": "evolved-continuum-ppo-training-receipt-v2",
        "registration_status": budget.registration_status,
        "registration_source": budget.registration_source,
        "data_role": budget.data_role,
        "root_seed": root_seed,
        "training_corpus_sha256": training.sha256,
        "validation_corpus_sha256": validation.sha256,
        "training_source_library_sha256": list(training.source_library_sha256),
        "validation_source_library_sha256": list(validation.source_library_sha256),
        "training_trajectory_sha256": list(training.trajectory_sha256s),
        "validation_trajectory_sha256": list(validation.trajectory_sha256s),
        "training_decision_count": training.decision_count,
        "validation_decision_count": validation.decision_count,
        "encoding": dataclasses.asdict(encoding),
        "node_features": list(NODE_FEATURES),
        "architecture_contract": ARCHITECTURE_CONTRACT,
        "reward_contract": REWARD_CONTRACT,
        "decision_step_unit": DECISION_STEP_UNIT,
        "max_total_timesteps": budget.max_total_timesteps,
        "actual_total_timesteps": sum(trial.total_timesteps for trial in budget.trials),
        "determinism": {"torch_deterministic_algorithms": True, "torch_threads": 1},
        "trials": trials,
        "champion_trial_index": champion["trial_index"],
        "champion_artifact": champion["artifact"],
        "champion_artifact_sha256": champion["artifact_sha256"],
    }
    receipt = {**payload, "sha256": _sha(payload)}
    receipt_path.write_text(_canonical(receipt) + "\n")
    write_checkpoint(trials, status="complete")
    return json.loads(_canonical(receipt))


_RECEIPT_KEYS = {
    "schema",
    "registration_status",
    "registration_source",
    "data_role",
    "root_seed",
    "training_corpus_sha256",
    "validation_corpus_sha256",
    "training_source_library_sha256",
    "validation_source_library_sha256",
    "training_trajectory_sha256",
    "validation_trajectory_sha256",
    "training_decision_count",
    "validation_decision_count",
    "encoding",
    "node_features",
    "architecture_contract",
    "reward_contract",
    "decision_step_unit",
    "max_total_timesteps",
    "actual_total_timesteps",
    "determinism",
    "trials",
    "champion_trial_index",
    "champion_artifact",
    "champion_artifact_sha256",
    "sha256",
}
_TRIAL_KEYS = {
    "trial_index",
    "seed",
    "derived_seed",
    "hyperparameters",
    "mean_validation_reward",
    "artifact",
    "artifact_sha256",
}


def load_frozen_ppo_bundle(
    directory: str | Path,
    *,
    expected_receipt_sha256: str,
    require_p2_frozen: bool = True,
    expected_data_role: str = "quarantined-pilot",
    nonconfirmatory_status: str = "development-provisional",
) -> PolicyBundleAdapter:
    _require_digest("expected_receipt_sha256", expected_receipt_sha256)
    root = Path(directory).resolve()
    receipt_path = root / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_KEYS:
        raise ValueError("PPO training receipt has an unknown or incomplete schema")
    if receipt["schema"] != "evolved-continuum-ppo-training-receipt-v2":
        raise ValueError("unknown PPO training receipt schema")
    claimed = receipt["sha256"]
    _require_digest("receipt sha256", claimed)
    if claimed != expected_receipt_sha256:
        raise ValueError("PPO receipt does not match the externally pinned digest")
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    if _sha(body) != claimed:
        raise ValueError("PPO training receipt digest mismatch")
    if receipt["registration_status"] not in {"development-provisional", "p2-frozen"}:
        raise ValueError("PPO receipt has an invalid registration status")
    if not isinstance(receipt["registration_source"], str) or not receipt["registration_source"]:
        raise ValueError("PPO receipt lacks a registration source")
    if (
        not isinstance(receipt["root_seed"], int)
        or isinstance(receipt["root_seed"], bool)
        or not 0 <= receipt["root_seed"] < 2**63
    ):
        raise ValueError("PPO receipt has an invalid root seed")
    for label in ("training_corpus_sha256", "validation_corpus_sha256"):
        _require_digest(label, receipt[label])
    if receipt["training_corpus_sha256"] == receipt["validation_corpus_sha256"]:
        raise ValueError("PPO receipt train/validation corpora are not disjoint")
    library_sets = []
    for label in (
        "training_source_library_sha256",
        "validation_source_library_sha256",
    ):
        values = receipt[label]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"PPO receipt {label} must be a non-empty unique list")
        for digest in values:
            _require_digest(label, digest)
        library_sets.append(set(values))
    if library_sets[0] & library_sets[1]:
        raise ValueError("PPO receipt train/validation libraries are not disjoint")
    trajectory_sets = []
    for label in ("training_trajectory_sha256", "validation_trajectory_sha256"):
        values = receipt[label]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"PPO receipt {label} must be a non-empty unique list")
        for digest in values:
            _require_digest(label, digest)
        trajectory_sets.append(set(values))
    if trajectory_sets[0] & trajectory_sets[1]:
        raise ValueError("PPO receipt train/validation trajectory content overlaps")
    for label in ("training_decision_count", "validation_decision_count"):
        value = receipt[label]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"PPO receipt {label} must be a positive int")
    if expected_data_role not in {"quarantined-pilot", "online-history-before-step"}:
        raise ValueError("invalid expected PPO data role")
    if receipt["data_role"] != expected_data_role:
        raise ValueError("PPO receipt has the wrong data role")
    if require_p2_frozen and expected_data_role != "quarantined-pilot":
        raise ValueError("confirmatory PPO must come from quarantined-pilot data")
    if require_p2_frozen and receipt["registration_status"] != "p2-frozen":
        raise ValueError("PPO receipt is not P2-frozen")
    if (
        receipt["architecture_contract"] != ARCHITECTURE_CONTRACT
        or receipt["reward_contract"] != REWARD_CONTRACT
    ):
        raise ValueError("PPO receipt changed the registered model/reward contract")
    if receipt["decision_step_unit"] != DECISION_STEP_UNIT:
        raise ValueError("PPO receipt changed the decision-step unit")
    if receipt["node_features"] != list(NODE_FEATURES):
        raise ValueError("PPO receipt changed the registered observation features")
    trials = receipt["trials"]
    if (
        not isinstance(trials, list)
        or not trials
        or any(not isinstance(trial, dict) or set(trial) != _TRIAL_KEYS for trial in trials)
    ):
        raise ValueError("PPO receipt contains malformed trials")
    if [trial["trial_index"] for trial in trials] != list(range(len(trials))):
        raise ValueError("PPO receipt trial indices are not contiguous")
    parsed_hyperparameters = []
    for trial in trials:
        if (
            not isinstance(trial["seed"], int)
            or isinstance(trial["seed"], bool)
            or not 0 <= trial["seed"] < 2**32
            or not isinstance(trial["derived_seed"], int)
            or isinstance(trial["derived_seed"], bool)
            or not 0 <= trial["derived_seed"] < 2**63
            or not isinstance(trial["mean_validation_reward"], int | float)
            or isinstance(trial["mean_validation_reward"], bool)
            or not math.isfinite(float(trial["mean_validation_reward"]))
        ):
            raise ValueError("PPO receipt trial contains invalid seed/reward fields")
        hyperparameters = trial["hyperparameters"]
        if not isinstance(hyperparameters, dict) or set(hyperparameters) != {
            field.name for field in dataclasses.fields(PPOHyperparameters)
        }:
            raise ValueError("PPO receipt trial changed the hyperparameter schema")
        parsed_hyperparameters.append(
            PPOHyperparameters(
                **{
                    **hyperparameters,
                    "hidden_sizes": tuple(hyperparameters["hidden_sizes"]),
                }
            )
        )
        artifact_name = trial["artifact"]
        if (
            not isinstance(artifact_name, str)
            or not artifact_name
            or Path(artifact_name).name != artifact_name
        ):
            raise ValueError("PPO trial artifact must be a safe basename")
        _require_digest("PPO trial artifact_sha256", trial["artifact_sha256"])
        trial_artifact = (root / artifact_name).resolve()
        if (
            trial_artifact.parent != root
            or not trial_artifact.is_file()
            or _sha256_file(trial_artifact) != trial["artifact_sha256"]
        ):
            raise ValueError("PPO trial artifact is missing or has the wrong digest")
    maximum = receipt["max_total_timesteps"]
    actual = receipt["actual_total_timesteps"]
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum < 1
        or not isinstance(actual, int)
        or isinstance(actual, bool)
        or actual != sum(trial.total_timesteps for trial in parsed_hyperparameters)
        or actual > maximum
    ):
        raise ValueError("PPO receipt timestep accounting is invalid")
    if receipt["determinism"] != {
        "torch_deterministic_algorithms": True,
        "torch_threads": 1,
    }:
        raise ValueError("PPO receipt changed the deterministic execution contract")
    champion_index = receipt["champion_trial_index"]
    if (
        not isinstance(champion_index, int)
        or isinstance(champion_index, bool)
        or not 0 <= champion_index < len(trials)
    ):
        raise ValueError("PPO receipt champion index is invalid")
    champion = trials[champion_index]
    expected_champion = max(
        trials,
        key=lambda trial: (
            trial["mean_validation_reward"],
            -trial["trial_index"],
        ),
    )
    if champion is not expected_champion:
        raise ValueError("PPO receipt champion is not the validation-best trial")
    if (
        receipt["champion_artifact"] != champion["artifact"]
        or receipt["champion_artifact_sha256"] != champion["artifact_sha256"]
    ):
        raise ValueError("PPO receipt champion fields disagree with its trial")
    artifact_name = receipt["champion_artifact"]
    if (
        not isinstance(artifact_name, str)
        or not artifact_name
        or Path(artifact_name).name != artifact_name
    ):
        raise ValueError("PPO champion artifact must be a safe basename")
    artifact = (root / artifact_name).resolve()
    if artifact.parent != root or not artifact.is_file():
        raise ValueError("PPO champion artifact escapes or is missing")
    _require_digest("champion artifact sha256", receipt["champion_artifact_sha256"])
    if _sha256_file(artifact) != receipt["champion_artifact_sha256"]:
        raise ValueError("PPO champion artifact digest mismatch")
    model = StableBaselinesPPOModel(artifact)
    if nonconfirmatory_status not in {"development-provisional", "exploratory-only"}:
        raise ValueError("invalid nonconfirmatory PPO bundle status")
    status = "p2-frozen" if require_p2_frozen else nonconfirmatory_status
    return frozen_ppo_bundle(
        model,
        GraphEncodingConfig(**receipt["encoding"]),
        training_receipt_sha256=claimed,
        registration_status=status,
    )


__all__ = [
    "ARCHITECTURE_CONTRACT",
    "ActionLogitModel",
    "DECISION_STEP_UNIT",
    "FrozenPPOProposer",
    "FrozenPPOScorer",
    "GraphEncodingConfig",
    "GraphObservationEncoder",
    "NODE_FEATURES",
    "PPOCorpus",
    "PPOHyperparameters",
    "PPOTuningBudget",
    "PlacementEpisode",
    "REWARD_CONTRACT",
    "StableBaselinesPPOModel",
    "corpus_from_snapshot_library",
    "frozen_ppo_bundle",
    "load_frozen_ppo_bundle",
    "load_tuning_budget",
    "tune_and_train_frozen_ppo",
]
