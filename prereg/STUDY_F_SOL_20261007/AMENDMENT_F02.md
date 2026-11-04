**Study F amendment F-02: second path of the Codex cleanup race — 7 October 2026**

**What happened.** After F-01 the run resumed at about 09:05 UTC. At 09:36 UTC the F-01 guard
stopped root T00's Sol arm on a harness fault (`PermissionError: [Errno 1] Operation not
permitted`, after 18 settled calls); as designed, the arm was set aside unchanged
(`T00/superseded_harness_0/`) and rerun from scratch. The F-01 wrapper had treated a refused
`killpg` as benign only when the CLI's leader process had already exited; this fault took
another timing path (the operator does not retain tracebacks, so the exact line is unknown).
A separate test showed that a live leader with an exited child does not trigger EPERM.

**Fix (Study F launcher only; `evolution/` unchanged).** When `killpg` is refused, the leader is
signalled directly (terminate, then kill, each followed by up to 2 s of waiting) and reaped; the
error is re-raised only if the leader still cannot be stopped. Harness faults now also record the
full traceback in the arm's `PAUSES.jsonl`. Tests: a live leader whose `killpg` is refused is
stopped directly; an unstoppable leader still raises.

**Disposition.** No change to treatment, selection, validation or analysis. The fix was swapped
in under this sealed amendment without stopping the launcher; Sol arms already running finish
on the F-01 code, and any harness fault there is set aside and rerun exactly as F-01 specifies.
Every set-aside is retained and reported. The decision depended only on the fault record, not on
any outcome.
