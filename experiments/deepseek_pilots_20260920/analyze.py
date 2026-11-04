"""Reconcile both resource pilots and project costs without reading scientific final grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.deepseek_pilots_20260920.client import LEGACY, legacy_exposure
from experiments.matched_transport_20260909.client import DurableOpenRouterClient

HERE = Path(__file__).parent
OUT = HERE / "artifacts"


def summarize():
    with (OUT / "journal.jsonl").open("rb") as stream:
        events = DurableOpenRouterClient._read(stream)
    completed = [e["payload"]["completion"] for e in events if e["payload"]["type"] == "completed"]
    if events[-1]["payload"]["type"] != "completed":
        raise ValueError("unresolved dispatch: do not project a successful campaign")
    with LEGACY.open("rb") as stream:
        prior = legacy_exposure(stream) / 1e6
    studies = {}
    files = {}
    for study in "AB":
        directory = OUT / study
        terminal = json.loads((directory / "COMPLETE.json").read_text())
        usage = json.loads((directory / "PROCESS_USAGE.json").read_text())
        if usage["exit_code"] != 0:
            raise ValueError("pilot process failed")
        records = [
            json.loads(line)
            for line in (directory / "campaign/events.jsonl").read_text().splitlines()
        ]
        rows = [r["payload"] for r in records if r["type"] == "candidate_completed"]
        initial = [r for r in rows if r["proposal_seq"] < 12]
        adaptive = [r for r in rows if r["proposal_seq"] >= 12]
        receipts = [json.loads(p.read_text()) for p in directory.glob("*-provider.json")]

        def work(rr):
            return sum(r["evaluator_work"]["candidate_microwork"] for r in rr) / 1e6

        initial_work, adaptive_work = work(initial), work(adaptive)
        if len(initial) != 12 or adaptive_work <= 0 or not receipts:
            raise ValueError("complete starting archive and measured adaptive work required")
        charges = [r["response"]["usage"]["cost"] for r in receipts]
        # Conservative pinned accounting, not cache-discounted charges, funds the next campaign.
        accounted = (
            sum(
                (
                    r["response"]["usage"]["prompt_tokens"] * 150000
                    + r["response"]["usage"]["completion_tokens"] * 600000
                    + 999999
                )
                // 1000000
                for r in receipts
            )
            / 1e6
        )
        resources = [
            json.loads(p.read_text()) for p in (directory / "evaluations").glob("*.resource.json")
        ]
        eval_seconds = sum(r["elapsed_seconds"] for r in resources)
        gen_seconds = sum(r["elapsed_seconds"] for r in receipts)
        projected = accounted * (56 - initial_work) / adaptive_work
        studies[study] = {
            "complete": True,
            "terminal": terminal,
            "initial_work": initial_work,
            "adaptive_work": adaptive_work,
            "adaptive_candidates": len(adaptive),
            "valid_adaptive_candidates": sum(bool(r["proposal"]["valid"]) for r in adaptive),
            "requests": len(receipts),
            "provider_charges_usd": sum(charges),
            "accounted_usd": accounted,
            "prompt_tokens": sum(r["response"]["usage"]["prompt_tokens"] for r in receipts),
            "completion_tokens": sum(r["response"]["usage"]["completion_tokens"] for r in receipts),
            "reasoning_tokens": sum(
                r["response"]["usage"]
                .get("completion_tokens_details", {})
                .get("reasoning_tokens", 0)
                for r in receipts
            ),
            "finish_reasons": [r["response"]["choices"][0].get("finish_reason") for r in receipts],
            "mean_request_seconds": gen_seconds / len(receipts),
            "peak_evaluation_rss_mib": max(r["peak_rss_bytes"] for r in resources) / 2**20,
            "projected_api_usd_per_pair": projected,
            "projected_serial_seconds_per_pair": 2
            * 56
            * eval_seconds
            / (initial_work + adaptive_work)
            + gen_seconds * (56 - initial_work) / adaptive_work,
            "observed_process_usage": usage,
        }
        for p in [
            directory / "COMPLETE.json",
            directory / "PROCESS_USAGE.json",
            directory / "campaign/events.jsonl",
            *directory.glob("*-provider.json"),
            *(directory / "evaluations").glob("*.resource.json"),
        ]:
            files[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    total = 30 * sum(s["projected_api_usd_per_pair"] for s in studies.values())
    pilot_accounted = sum(c["cost_microusd"] for c in completed) / 1e6
    reserved_total = prior + pilot_accounted + total * 1.4
    result = {
        "complete": True,
        "studies": studies,
        "prior_accounted_and_unresolved_usd": prior,
        "pilot_accounted_usd": pilot_accounted,
        "projection": {
            "roots_per_study": 30,
            "work_ceiling": 56,
            "api_usd": total,
            "project_total_with_40pct_reserve_usd": reserved_total,
            "fits_60_usd_with_reserve": reserved_total <= 60,
            "serial_hours": 30
            * sum(s["projected_serial_seconds_per_pair"] for s in studies.values())
            / 3600,
        },
        "limitations": [
            "One root and at most six requests per task; projection is not a worst-case bound",
            "GP evaluator mix is approximated by the LLM pilot mix",
            "Compute projection excludes final evaluation, scientific gates and analysis",
            "Provider price cap excludes expensive weekday periods; waiting time is not projected",
            "No policy-quality or operator-effect inference from these resource pilots",
        ],
        "source_files_sha256": files,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    with (OUT / "SUMMARY.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result["projection"], indent=2))


if __name__ == "__main__":
    summarize()
