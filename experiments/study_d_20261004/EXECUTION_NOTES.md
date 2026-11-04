**Study D execution notes (no protocol change)**

- 2026-10-04 07:59 UTC: launch under P2_FREEZE.json.
- 08:19 UTC: the host disk filled (about 3 GB free before launch; Study D's own runs were 189 MB).
  GP searches of D02 and D03 crashed with `OSError: [Errno 28] No space left on device` while
  writing an evaluation resource record; D02's LLM arm exited with an unresolved request
  (exit 40). The launcher logged `failed` and stopped new work; the investigator's agent then
  sent SIGTERM to the remaining stage children (D00, D01, D04, D05 LLM searches).
- All run JSON/JSONL files were checked and parse; no root was marked stopped.
- 14:44 UTC, after the investigator freed disk space (44 GB free): resumed with the unchanged
  launcher. Four in-flight requests (D01, D02, D03, D05) were reconciled by appended charged
  settlements of $0.05 each (P2 amendment 02 mechanism inherited via Study C); no journal event
  was edited and no root replaced. Interrupted arms resume from their durable campaign state;
  affected roots: D00–D05. They are reported separately as for studies A and B.
- Seven interrupted evaluations (D00 gp/llm, D01 gp, D02 gp, D03 gp, D04 llm, D05 gp) were moved
  aside by the inherited recovering evaluator (`interrupted-*` directories, retained) and
  re-requested. Checked for D02 gp seq 073: the re-run request has identical content (its file
  hash differs only by frozenset pickling order across processes), and its LadderResult equals
  the interrupted run's completed result.
