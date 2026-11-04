# Corrected operator comparison — preparation, not a frozen protocol

The September 9 audit proves that LLM-only batching changed initialization constraints and
parent-update cadence in the historical study. The corrected comparison will use one proposal
per call in **both** arms, retaining multiple islands and a common initialization archive.
This changes the experiment design; it does not repair or relabel historical trajectories.

Sequential scheduling removes the measured batch/island mismatch and provides the same
candidate-level opportunity to update parents in each arm. The choice is made from the control
audit, before any corrected comparative search outcome. Batched throughput experiments may be
reported separately, but cannot support operator isolation without matched GP sibling batches.

`control.py` rejects batched plans and mismatched initialization records. `runner.py` integrates
the control into the production campaign runner via the existing fixed-archive seam. It stops
before allocating candidate 12 (zero-based), then verifies an exclusive paired receipt before
every adaptive allocation and dispatch. The receipt binds the immutable initialization prefixes,
both manifests, effective seed grids and constraints, source hashes, evaluations, work charges,
parent selections and survivor decisions. Candidate IDs are compared by sequence because arm
names differ. Appending later candidates does not invalidate the initialization prefix.

Tests reject all 20 historical roots and exercise a positive paired run through the production
runner with a deterministic controlled evaluator: identical actual evaluator inputs, 12 equal
initial candidates, zero proposal calls before certification, successful continuation, byte-exact
interruption replay, and refusal before dispatch after peer tampering or accounting mismatch.
This controlled-evaluator test is an execution-control witness. The separate real-simulator
pilot is now complete (below); neither is a new comparative study.
The new CLI keeps the underlying campaign CLI's existing scientific and cost gates.

Use `python -m experiments.matched_control_20260909.run_campaign` with the ordinary campaign
arguments plus `--seed-archive PATH --initial PATH --pair-receipt PATH`. Exit code 10 means the
initialization checkpoint was reached; it does not mean campaign completion. Run both arms to
this checkpoint, then use the same module with `--pair-receipt PATH --certify-pair GP_DIR LLM_DIR`.
Resume each original campaign command afterward. Paths in this operational receipt are local;
an artifact release needs a separate portable manifest. A receipt establishes execution parity,
not study authorization, public registration, or source/environment closure.

Before launching:

1. Run the controlled-evaluator regression suite and a bounded real-simulator initialization
   pilot with the intended distinct starting policies, outside all study roots. Check resource
   bounds and replay of real evaluations. The current regression witness uses 12 copies of one
   policy to isolate scheduling; it does not establish the quality of the study's seed archive.
2. Exercise subprocess CLI orchestration, checkpoint/resume and API cost journals with a fake
   completion endpoint before dispatching paid calls; include the new seam in the source freeze.
3. Freeze scenario/calibration choices, strong simple baselines, root count and namespaces,
   evaluator-work budget, primary/secondary estimands, practical thresholds, multiplicity,
   stopping rules, model/API settings, and a bounded total cost before launch.
4. Preserve a complete executable source/environment snapshot and independent validation split.
5. Run a resource/cost pilot outside the study roots, then execute the frozen replication and
   report all outcomes. Evaluate simulator validity gates required by the retained claims.

The historical recovery/rescoring in `gp_tuning_20260902/validated_20260909/` can proceed because
it characterizes the historical systems. Its outcome cannot substitute for this corrected study.

`real_pilot.py` now prepares the distinct-policy simulator witness: the actual 12-member root-801
archive on a separate pilot seed grid, a GP initialization and a process-interrupted/resumed LLM
initialization, both with a forbidden proposal operator. It checks identical evaluator inputs and
complete initialization receipts under 2 GiB/1,200-second child limits, preserving sources and logs.
GPHH scoring and full closure are complete. Two pilot setup attempts stopped before evaluation:
the first compared tuple-containing package metadata with its JSON-decoded lists; the second
violated the campaign plan's ten-definition invariant. Both source snapshots and failure records
are preserved. Version 3 normalizes snapshots canonically and retains all ten plan definitions
while executing only the explicitly identified pilot root. Version 3 completed with 12 real
evaluations per arm, exact input/initialization parity, successful interruption recovery and
zero proposal calls. All 209 source/native archive members verify. See
`real_pilot_20260909_v3/REPORT.md`. It does not exercise adaptive search or the model API.

Corrected OpenRouter CLI runs now require `--proposal-journal PATH`, with `max_attempts: 1` in the
engine configuration. The separate durable transport replays saved responses and stops on uncertain
dispatch instead of silently retrying. See `../matched_transport_20260909/README.md`; nine transport
tests pass, including a local HTTPS endpoint across process restart. The full subprocess
CLI/campaign-ledger pilot also passes saved-response recovery and uncertain-dispatch refusal,
with zero initialization calls and matching transport/campaign costs. See
`../matched_cli_pilot_20260909/README.md`. Scenario, source, analysis and paid-cost freezes remain
required; these witnesses do not authorize a confirmatory campaign.
