"""One comparison boundary for every Phase-7 baseline.

The adapter deliberately contains no simulator shortcuts.  Scoring flows through the
same PreScore-like batch contract and rebalancing through the same fixed action gateway
as evolved PlacementScript bundles.  Full evaluation and ladder methods delegate to
the Phase-5 authorities rather than reimplementing C20 fitness or rung promotion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from evolution.evaluation import PolicyEvaluation, TaskEvaluationConfig, evaluate_policy_adapter
from evolution.ladder import (
    LadderConfig,
    LadderResult,
    evaluate_adapter_through_ladder,
)


@dataclass(frozen=True)
class PolicyBundleAdapter:
    """A versioned scorer + proposer with canonical, archive-safe metadata."""

    name: str
    version: str
    scorer: Any
    proposer: Any
    representation: str
    artifact_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (("name", self.name), ("version", self.version)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"baseline {label} must be a non-empty string")
        if not isinstance(self.representation, str) or not self.representation:
            raise ValueError("baseline representation must be a non-empty string")
        if len(self.artifact_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.artifact_sha256
        ):
            raise ValueError("baseline artifact_sha256 must be a lowercase SHA-256 digest")
        if not callable(getattr(self.scorer, "score_batch", None)):
            raise TypeError("baseline scorer must implement score_batch")
        if not callable(getattr(self.proposer, "propose_moves", None)):
            raise TypeError("baseline proposer must implement propose_moves")
        try:
            encoded = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise TypeError("baseline metadata must be canonical-JSON serializable") from exc
        object.__setattr__(self, "metadata", json.loads(encoded))

    @property
    def identity_sha256(self) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "representation": self.representation,
            "artifact_sha256": self.artifact_sha256,
            "metadata": self.metadata,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def evaluate(
        self,
        task: TaskEvaluationConfig,
        *,
        proposal_seq: int,
        require_complete_development_set: bool = True,
        config_transform: Any | None = None,
    ) -> PolicyEvaluation:
        return evaluate_policy_adapter(
            self.scorer,
            self.proposer,
            task,
            proposal_seq=proposal_seq,
            policy_name=self.name,
            policy_version=self.version,
            policy_identity_sha256=self.identity_sha256,
            policy_artifact_sha256=self.artifact_sha256,
            require_complete_development_set=require_complete_development_set,
            config_transform=config_transform,
        )

    def evaluate_through_ladder(
        self,
        task: TaskEvaluationConfig,
        ladder: LadderConfig,
        *,
        proposal_seq: int,
    ) -> LadderResult:
        return evaluate_adapter_through_ladder(
            self.scorer,
            self.proposer,
            task,
            ladder,
            proposal_seq=proposal_seq,
            policy_name=self.name,
            policy_version=self.version,
            policy_identity_sha256=self.identity_sha256,
            policy_artifact_sha256=self.artifact_sha256,
        )
