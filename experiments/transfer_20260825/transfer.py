"""Zero-shot transfer of evolved champions onto a cluster they never saw.

Why this endpoint exists
------------------------
Every 2026-08 comparison was decided by *best-of-run energy on the scenario the search
optimized*. That endpoint has two properties that made the whole study unaffordable and
uninformative at once: it is a maximum statistic (paired root SD 0.058 against effects
near 0.02, so `ESTIMAND_POWER.md` priced it at 63 roots for 90% power), and it rewards
whatever exploits the training instance, which is exactly what an unguided mutation
that deletes a normalization term does best.

Transfer asks a different question, and the one an operator actually cares about: does
the policy still work somewhere else? It costs **zero model calls** — champions are
already archived — and it is where a structured, readable policy should separate from a
lucky mutilation, because a policy that won by breaking an invariant on one topology has
no reason to survive a different one.

What it reports
---------------
For each arm's champion, the source (training) fitness, the target fitness, and the
**transfer gap**. Feasibility is reported as a first-class outcome: a champion that is
infeasible on the target has failed transfer regardless of its energy, and that is the
headline number, not a footnote.

Nothing here is confirmatory. It is a development endpoint on quarantined scenarios.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TransferOutcome:
    arm: str
    run: str
    policy_sha256: str
    source_feasible: bool
    source_energy: float | None
    target_feasible: bool
    target_energy: float | None
    target_worst_slo: float

    @property
    def transfer_gap(self) -> float | None:
        """Target minus source normalized energy; None when either side is infeasible."""
        if self.source_energy is None or self.target_energy is None:
            return None
        return self.target_energy - self.source_energy

    @property
    def survived(self) -> bool:
        return self.source_feasible and self.target_feasible

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["transfer_gap"] = self.transfer_gap
        payload["survived"] = self.survived
        return payload


def _load_task(config: str | Path, seeds: int | None = None):
    from evolution.evaluation import ScenarioGrid, load_task_evaluation_config

    task = load_task_evaluation_config(ROOT / config)
    if seeds is not None:
        task = dataclasses.replace(
            task,
            scenarios=tuple(
                ScenarioGrid(item.scenario_id, item.path, item.seeds[:seeds])
                for item in task.scenarios
            ),
        )
    return task


# Archive directory name per arm; the other arms are evaluation-only references or do
# not archive PlacementScript sources (classic GPHH evolves DEAP trees).
_ARCHIVE_DIR = {"llm": "llm", "same_dsl_gp": "same_dsl_gp", "random_ast": "random_ast"}


def champions_from_run(run_dir: Path) -> dict[str, str]:
    """Best archived PlacementScript source per arm, read from the sealed run report.

    Champion records reference the hash-chained archive rather than inlining source
    (the 2026-08 artifact-size fix), so the source is read back from
    ``archives/<arm>/candidates/<proposal_seq>.ps`` and its digest re-derived.
    """
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return {}
    body = json.loads(report_path.read_text())
    body = body.get("payload", body)
    out: dict[str, str] = {}
    for arm, archive_name in _ARCHIVE_DIR.items():
        arm_body = body.get("arms", {}).get(arm)
        if not isinstance(arm_body, dict):
            continue
        # `best_absolute` is a float (the energy); the authoritative pointer to the
        # champion is `best_found_at`, with `best` as a secondary carrier of the same
        # proposal_seq. Reading the float as a record silently falls through to the
        # trajectory scan below, which happens to agree -- but only by accident.
        seq = None
        for key in ("best_found_at", "best"):
            record = arm_body.get(key)
            if isinstance(record, dict) and record.get("proposal_seq") is not None:
                seq = record["proposal_seq"]
                break
        if seq is None:
            # Fall back to the cheapest feasible trajectory row.
            rows = [
                row
                for row in arm_body.get("trajectory", [])
                if isinstance(row.get("candidate_normalized_energy"), (int, float))
            ]
            if not rows:
                continue
            seq = min(rows, key=lambda row: row["candidate_normalized_energy"])["proposal_seq"]
        path = run_dir / "archives" / archive_name / "candidates" / f"{int(seq):08d}.ps"
        if not path.is_file():
            continue
        out[arm] = path.read_text(encoding="utf-8")
    return out


def evaluate_transfer(
    champions: dict[str, str],
    *,
    run_label: str,
    source_config: str,
    target_config: str,
    seeds: int | None = None,
) -> tuple[TransferOutcome, ...]:
    from evolution.evaluation import evaluate_policy_source

    source_task = _load_task(source_config, seeds)
    target_task = _load_task(target_config, seeds)

    out: list[TransferOutcome] = []
    for index, (arm, source) in enumerate(sorted(champions.items())):
        digest = hashlib.sha256(source.encode()).hexdigest()
        try:
            on_source = evaluate_policy_source(source, source_task, proposal_seq=950000 + index)
            on_target = evaluate_policy_source(source, target_task, proposal_seq=960000 + index)
        except Exception:  # noqa: BLE001 - a champion that cannot run has not transferred
            continue
        worst = min(
            (min(item.slo_achieved.values()) for item in on_target.instances), default=1.0
        )
        out.append(
            TransferOutcome(
                arm=arm,
                run=run_label,
                policy_sha256=digest,
                source_feasible=on_source.fitness.feasible,
                source_energy=on_source.fitness.normalized_energy,
                target_feasible=on_target.fitness.feasible,
                target_energy=on_target.fitness.normalized_energy,
                target_worst_slo=worst,
            )
        )
    return tuple(out)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seal_payload(payload: Any) -> dict[str, Any]:
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
    }
