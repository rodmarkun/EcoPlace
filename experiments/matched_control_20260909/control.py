"""Fail-closed prospective control contract for sequential operator comparisons.

Historical batched results remain untouched. Batched comparisons require their own
matched sibling/parent-update implementation and are not certified by this contract.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequentialControl:
    proposals_per_call: int = 1
    initialization_size: int = 12

    def validate_plan(self, plan) -> None:
        if self.proposals_per_call != 1 or plan.proposals_per_call != 1:
            raise ValueError('operator isolation requires sequential proposals in both arms')
        if plan.candidate_budget < self.initialization_size:
            raise ValueError('candidate budget cannot truncate the common initialization')

    def validate_initialization(self, left: list[dict], right: list[dict]) -> None:
        if len(left) != self.initialization_size or len(right) != self.initialization_size:
            raise ValueError('paired initialization is incomplete')
        for seq, (a, b) in enumerate(zip(left, right, strict=True)):
            if a['proposal_seq'] != seq or b['proposal_seq'] != seq:
                raise ValueError('paired initialization order differs')
            for field in ('source_sha256', 'island_id', 'fitness', 'instance_metrics'):
                if a[field] != b[field]:
                    raise ValueError(f'paired initialization differs at {seq}: {field}')
