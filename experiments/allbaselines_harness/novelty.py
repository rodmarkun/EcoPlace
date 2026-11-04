"""Enhanced-profile novelty evaluator shared by non-CampaignRunner adaptive arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analysis.tracking import derive_seed
from evolution.ladder import (
    BehavioralNoveltyConfig,
    LadderResult,
    evaluate_adapter_through_ladder,
    evaluate_through_ladder,
)


@dataclass
class AdaptiveNoveltyEvaluator:
    root_seed: int
    tolerance: float
    known_signatures: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.config = BehavioralNoveltyConfig(
            tolerance=self.tolerance,
            signature_seed=derive_seed(
                self.root_seed,
                "evolution",
                "behavioral-novelty-signature",
            ),
        )

    def _record(self, result: LadderResult) -> LadderResult:
        for trace in result.trace:
            signature = trace.get("behavior_signature")
            if isinstance(signature, dict) and isinstance(signature.get("sha256"), str):
                self.known_signatures.add(signature["sha256"])
        return result

    def evaluate_source(self, source: str, task: Any, ladder: Any, proposal_seq: int):
        return self._record(
            evaluate_through_ladder(
                source,
                task,
                ladder,
                proposal_seq=proposal_seq,
                behavioral_novelty=self.config,
                known_l1_signatures=frozenset(self.known_signatures),
            )
        )

    def evaluate_bundle(self, bundle: Any, task: Any, ladder: Any, proposal_seq: int):
        return self._record(
            evaluate_adapter_through_ladder(
                bundle.scorer,
                bundle.proposer,
                task,
                ladder,
                proposal_seq=proposal_seq,
                policy_name=bundle.name,
                policy_version=bundle.version,
                policy_identity_sha256=bundle.identity_sha256,
                policy_artifact_sha256=bundle.artifact_sha256,
                behavioral_novelty=self.config,
                known_l1_signatures=frozenset(self.known_signatures),
            )
        )


__all__ = ["AdaptiveNoveltyEvaluator"]
