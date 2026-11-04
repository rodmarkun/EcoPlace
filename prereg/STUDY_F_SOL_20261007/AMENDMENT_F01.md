**Study F amendment F-01: Codex process-cleanup race; Sol arms set aside and rerun — 7 October 2026**

**What happened.** The scientific run started at 08:19 UTC. At about 08:50 UTC the investigator's
monitoring found that two Sol arms (W02, T02) had stopped receiving proposals. In each, a Codex
call completed normally (`turn.completed` in the usage journal), and the harness then raised
`PermissionError: [Errno 1] Operation not permitted` while stopping the finished CLI process.
The reservation was never settled, so every later attempt failed before dispatch ("Codex usage
ledger already has a pending dispatch"): W02 recorded 550 such candidates and ended at the
candidate safety ceiling after 8.8 of 56 work units with zero settled calls; T02 had 70 after 9
successful calls. Sol never saw those prompts; these are harness faults, not treatment outcomes.

**Cause (reproduced).** `evolution/codex_engine._stop_process_group` calls `os.killpg` and
tolerates only `ProcessLookupError`. On macOS, `killpg` on a group whose leader has exited but is
not yet reaped raises `EPERM` (reproduced with `sh -c 'exit 0'`; after reaping it raises
`ProcessLookupError`). The Codex CLI often exits right after its final event, racing the cleanup.
The smoke (10 calls) did not hit it; the run hit it in 2 of ~110 calls.

**Fix (Study F launcher only; `evolution/` unchanged, so the environment digest still equals E's).**
1. `_stop_process_group` is wrapped: a `PermissionError` is treated as "already exited" only when
   the leader has in fact exited (then reaped); otherwise it is re-raised.
2. Any other host-side failure inside a Sol call stops the arm with exit 70 (`HarnessStop`) and is
   logged to the arm's `PAUSES.jsonl`, instead of letting an unsettled reservation fail later
   attempts.
3. The launcher sets a stopped arm aside unchanged (`<root>/superseded_harness_<n>/`) and reruns it
   from scratch at once, under identical treatment and within the existing limit of 6 set-asides
   per root (shared with quota set-asides).
Regression tests: `test_launch.py` (real EPERM reproduction; live leader still raises; host
failure → exit 70; set-aside).

**Disposition, decided before any selection, validation of a Sol arm, or final grid was
inspected.** To avoid harness-caused abandoned attempts in arms killed mid-call, **every Sol arm
started before this amendment (R00–R02, T00–T03, W00–W03; 11 roots) is set aside unchanged** to
`<root>/superseded_F01/` (arm, pair receipt, exit record, and any Sol validation) and rerun from
scratch. The rule depends only on whether an arm had started, not on its state or outcome. No
Sol selection was sealed and no final grid was run before the stop. Validation records of E's
GP and DeepSeek archives are deterministic and are kept. The stop was made by terminating the
launcher and its children at about 08:55 UTC.
