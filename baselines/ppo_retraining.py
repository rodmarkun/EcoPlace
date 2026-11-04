"""Exploratory periodic PPO retraining with strict pre-trigger provenance (TODO 7.7)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.tracking import derive_seed
from baselines.adapter import PolicyBundleAdapter
from baselines.ppo import (
    ARCHITECTURE_CONTRACT,
    DECISION_STEP_UNIT,
    REWARD_CONTRACT,
    GraphEncodingConfig,
    PPOCorpus,
    PPOHyperparameters,
    PPOTuningBudget,
    load_frozen_ppo_bundle,
    tune_and_train_frozen_ppo,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class PeriodicPPOPlan:
    status: str
    registration_source: str
    interval_steps: int
    max_retrains: int
    max_total_timesteps: int
    step_unit: str
    hyperparameters: PPOHyperparameters
    encoding: GraphEncodingConfig

    def __post_init__(self) -> None:
        if self.status != "exploratory-only":
            raise ValueError("periodic PPO retraining is structurally exploratory-only")
        if not self.registration_source:
            raise ValueError("periodic PPO registration_source is required")
        if self.step_unit != DECISION_STEP_UNIT:
            raise ValueError("periodic PPO must use placement-decision steps")
        for name in ("interval_steps", "max_retrains", "max_total_timesteps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive int")
        if self.max_retrains * self.hyperparameters.total_timesteps > self.max_total_timesteps:
            raise ValueError("periodic PPO schedule exceeds max_total_timesteps")

    @property
    def sha256(self) -> str:
        return _sha(dataclasses.asdict(self))

    def training_budget(self) -> PPOTuningBudget:
        return PPOTuningBudget(
            registration_status="development-provisional",
            registration_source=self.registration_source,
            data_role="online-history-before-step",
            trials=(self.hyperparameters,),
            max_total_timesteps=self.hyperparameters.total_timesteps,
        )


def load_periodic_ppo_plan(path: str | Path) -> PeriodicPPOPlan:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != "evolved-continuum-periodic-ppo-v1":
        raise ValueError("unknown periodic-PPO schema")
    if (
        raw.get("architecture_contract") != ARCHITECTURE_CONTRACT
        or raw.get("reward_contract") != REWARD_CONTRACT
    ):
        raise ValueError("periodic PPO file changed the model/reward contract")
    return PeriodicPPOPlan(
        status=raw["status"],
        registration_source=raw["registration_source"],
        interval_steps=raw["interval_steps"],
        max_retrains=raw["max_retrains"],
        max_total_timesteps=raw["max_total_timesteps"],
        step_unit=raw["step_unit"],
        hyperparameters=PPOHyperparameters(
            **{
                **raw["hyperparameters"],
                "hidden_sizes": tuple(raw["hyperparameters"]["hidden_sizes"]),
            }
        ),
        encoding=GraphEncodingConfig(**raw["encoding"]),
    )


@dataclass(frozen=True)
class RetrainingBatch:
    step: int
    training: PPOCorpus
    validation: PPOCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step must be a non-negative int")
        if (
            self.training.data_role != "online-history-before-step"
            or self.validation.data_role != "online-history-before-step"
        ):
            raise ValueError("PPO retraining corpora have an invalid data role")
        if self.training.split_role != "train" or self.validation.split_role != "validation":
            raise ValueError("PPO retraining requires train and validation splits")
        episodes = (*self.training.episodes, *self.validation.episodes)
        if any(episode.decision_end_step >= self.step for episode in episodes):
            raise ValueError("PPO retraining history must end strictly before its step")

    @property
    def history_end_step(self) -> int:
        return max(
            episode.decision_end_step
            for episode in (*self.training.episodes, *self.validation.episodes)
        )


PPOTrainFn = Callable[
    [PPOCorpus, PPOCorpus, PPOTuningBudget, GraphEncodingConfig, Path, int],
    dict[str, Any],
]


def _default_train(
    training: PPOCorpus,
    validation: PPOCorpus,
    budget: PPOTuningBudget,
    encoding: GraphEncodingConfig,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    return tune_and_train_frozen_ppo(training, validation, budget, encoding, output, root_seed=seed)


@dataclass(frozen=True)
class PeriodicPPOEvent:
    retrain_index: int
    step: int
    history_end_step: int
    seed: int
    training_corpus_sha256: str
    validation_corpus_sha256: str
    training_receipt: dict[str, Any]
    artifact_directory: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class PeriodicPPORun:
    root_seed: int
    plan_sha256: str
    output_root: str
    events: tuple[PeriodicPPOEvent, ...]

    @property
    def total_timesteps(self) -> int:
        return sum(event.training_receipt["actual_total_timesteps"] for event in self.events)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "evolved-continuum-periodic-ppo-run-v2",
            "role": "exploratory-periodic-retraining",
            "root_seed": self.root_seed,
            "plan_sha256": self.plan_sha256,
            "output_root": self.output_root,
            "retrain_count": len(self.events),
            "total_timesteps": self.total_timesteps,
            "events": [event.to_dict() for event in self.events],
        }
        return {**payload, "sha256": _sha(payload)}

    def load_final_bundle(self) -> PolicyBundleAdapter:
        if not self.events:
            raise ValueError("periodic PPO run contains no trained policy")
        event = self.events[-1]
        return load_frozen_ppo_bundle(
            event.artifact_directory,
            expected_receipt_sha256=event.training_receipt["sha256"],
            require_p2_frozen=False,
            expected_data_role="online-history-before-step",
            nonconfirmatory_status="exploratory-only",
        )


class PeriodicPPOArm:
    def __init__(self, plan: PeriodicPPOPlan, *, train_fn: PPOTrainFn = _default_train) -> None:
        self.plan = plan
        self.train_fn = train_fn

    def run(
        self, batches: tuple[RetrainingBatch, ...], output_dir: str | Path, *, root_seed: int
    ) -> PeriodicPPORun:
        if (
            not isinstance(root_seed, int)
            or isinstance(root_seed, bool)
            or not 0 <= root_seed < 2**63
        ):
            raise ValueError("root_seed must be an explicit integer in [0, 2**63)")
        if len(batches) > self.plan.max_retrains:
            raise ValueError("periodic PPO batches exceed max_retrains")
        steps = [batch.step for batch in batches]
        if steps != sorted(set(steps)):
            raise ValueError("periodic PPO batch steps must be unique and increasing")
        if any(step == 0 or step % self.plan.interval_steps != 0 for step in steps):
            raise ValueError("PPO retraining batch is outside the registered schedule")
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        budget = self.plan.training_budget()
        events = []
        for retrain_index, batch in enumerate(batches):
            seed = derive_seed(
                root_seed, "periodic-ppo", self.plan.sha256, retrain_index, batch.step
            )
            artifact_directory = output / f"retrain-{retrain_index:03d}"
            receipt = self.train_fn(
                batch.training,
                batch.validation,
                budget,
                self.plan.encoding,
                artifact_directory,
                seed,
            )
            if (
                receipt.get("data_role") != "online-history-before-step"
                or receipt.get("root_seed") != seed
            ):
                raise ValueError("PPO retraining receipt changed provenance or seed")
            if (
                receipt.get("training_corpus_sha256") != batch.training.sha256
                or receipt.get("validation_corpus_sha256") != batch.validation.sha256
            ):
                raise ValueError("PPO retraining receipt changed the sealed corpus")
            if receipt.get("actual_total_timesteps") != self.plan.hyperparameters.total_timesteps:
                raise ValueError("PPO retraining did not consume its exact fixed budget")
            trials = receipt.get("trials")
            if (
                not isinstance(trials, list)
                or len(trials) != 1
                or _canonical(trials[0].get("hyperparameters"))
                != _canonical(dataclasses.asdict(self.plan.hyperparameters))
            ):
                raise ValueError("PPO retraining receipt changed fixed hyperparameters")
            digest = receipt.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("PPO retraining receipt lacks a SHA-256 digest")
            if _sha({key: value for key, value in receipt.items() if key != "sha256"}) != digest:
                raise ValueError("PPO retraining receipt digest mismatch")
            artifact_name = receipt.get("champion_artifact")
            artifact_digest = receipt.get("champion_artifact_sha256")
            if not isinstance(artifact_name, str) or Path(artifact_name).name != artifact_name:
                raise ValueError("PPO retraining receipt lacks a safe champion artifact")
            artifact = (artifact_directory / artifact_name).resolve()
            if (
                artifact.parent != artifact_directory.resolve()
                or not artifact.is_file()
                or hashlib.sha256(artifact.read_bytes()).hexdigest() != artifact_digest
            ):
                raise ValueError("PPO retraining champion artifact digest mismatch")
            events.append(
                PeriodicPPOEvent(
                    retrain_index,
                    batch.step,
                    batch.history_end_step,
                    seed,
                    batch.training.sha256,
                    batch.validation.sha256,
                    json.loads(_canonical(receipt)),
                    str(artifact_directory),
                )
            )
        run = PeriodicPPORun(root_seed, self.plan.sha256, str(output), tuple(events))
        if run.total_timesteps > self.plan.max_total_timesteps:
            raise ValueError("periodic PPO run exceeded max_total_timesteps")
        return run


__all__ = [
    "PeriodicPPOArm",
    "PeriodicPPOEvent",
    "PeriodicPPOPlan",
    "PeriodicPPORun",
    "RetrainingBatch",
    "load_periodic_ppo_plan",
]
