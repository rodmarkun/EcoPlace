**Study F: Study E's roots with a stronger proposal engine and validation re-selection — internally frozen protocol, 7 October 2026**

**Status.** Frozen by the investigator before any scientific root was run. Pilot values below
come from the smoke records (`experiments/study_f_20261007/smoke/`), which are committed with
the freeze; smoke outcomes are not analysed.

**Motivation, disclosed (read first).** Study E (registered, sealed and analysed before this
protocol was written) did not support its primary: LLM champion versus GP champion over 90 roots,
27–39–24, p = 0.18, direction favouring GP. Study F is designed **after E's outcomes were known
and is motivated by E's null**. It asks two questions: does a stronger proposal engine beat E's
engine on E's roots, and does it then beat GP? The following commitments are part of the
registration:
- **One pre-specified engine.** F tests a single engine package (below), chosen before any F
  run. The investigator will report F whatever its result, and will report any further engine
  studies on these roots alongside F, together with their number.
- **Study E stands as registered.** F does not amend, replace or reinterpret E; E's primary
  remains "not supported" and is reported as such in every write-up that reports F.
- **Known outcomes.** The investigator knows E's per-root outcomes, including the reused arms'
  final-grid scores. No F design choice (roots, endpoints, engine settings, validation rule,
  analysis) conditions on per-root outcomes: F uses all 90 of E's roots unchanged. The validation
  step and the endpoint order were motivated by E's aggregate results (disclosed below).
- **Reviewer overlap.** The Sol model family (via Codex review sessions) served as the project's
  design reviewer, including for E. In F it is also the proposer. Proposer calls run in an empty
  isolated directory and cannot read the repository, the protocol, E's results or any review.
- **No prior exposure.** No earlier Codex pilot used these roots.
- **Related variants.** As in E, the three families are related variants of one sealed generator
  (`sim/scenarios/dev_tight.py`): T and R share the cluster and workload, W shares the generator
  and its calibration machinery. Breadth claims are limited accordingly.

**Status under Researcher Amendment A1.** A1 (`ContinuumPlacementScopedDown.md`) allows no
confirmatory number from the subscription engine under the current registration unless "a new
registration version … re-defines the confirmatory LLM arm around a commercial engine". **This
protocol is that new registration version** for Study F's Sol arm. Its threats are stated
plainly: the engine is a commercial subscription model whose weights, serving stack and future
availability are outside the investigator's control, so live re-runs are not reproducible. In
place of weight hashes F records, for every call, the CLI version, the requested model and
reasoning effort, and every prompt and response, so that search is re-executable by
archived-response replay (A1 item 3). **The Codex CLI's output stream does not report which
model served a call** (checked in the smoke: it echoes only the requested model), so the served
model cannot be verified or pinned; this is disclosed as a threat, not mitigated. Per-token energy is not computable for this engine (A1
item 4); CLI-reported token counts are retained descriptively.

**Hypotheses.** Under validation re-selection and otherwise Study E's package:
(HS) the Sol engine finds better placement policies than E's DeepSeek engine;
(HP) the Sol engine finds better placement policies than E's same-representation GP.

**Treatment: engine-package swap.** The contrast is **"Sol, medium reasoning, via the Codex CLI"
versus "DeepSeek V4.1 Flash, low reasoning, via OpenRouter"** (E's `engine.json`): model,
reasoning effort, provider, client and sampling all change together. It is not a pure model swap.
- Model `gpt-6.1-sol` through the Codex CLI subscription, codex-cli **0.159.1** (pinned),
  `model_reasoning_effort=medium`, per-call timeout 600 s. **Effort, disclosed:** the smoke ran
  at `high` (10 calls, median ~100 s). To fit the investigator's one-day wall-time budget on the
  16 GB host, the investigator switched to `medium` before the freeze and before any scientific
  root (one real E rewrite prompt: 156 s at high, 54 s at medium). The choice was made on wall
  time alone, without any F outcome.
- Each call runs in a fresh, isolated, empty temporary directory, read-only sandbox, ephemeral
  session, user configuration ignored, private ephemeral authentication state; no repository,
  tool output or prior session is visible to the model.
- Disclosed differences: sampling (temperature, top-p) is managed by the subscription and cannot
  be set (E: 0.7, 0.8); Codex exposes no output-token limit (E: 32,000); no provider seed; the
  prompts are E's DeepSeek-era prompts unchanged, and Codex wraps them in its own agent
  harness and system prompt.
- **Codex failures are treatment outcomes.** A timeout, non-zero exit or unparseable response
  fails its attempt and consumes attempts exactly as the inherited operator rules do, with the
  engine configuration's circuit breaker of 3 consecutive failures.
- **Subscription refusals are not treatment outcomes.** On a rate- or usage-limit message the
  arm is stopped, set aside unchanged to `<root>/superseded_quota_<n>/` (retained, never edited)
  and rerun from scratch under identical treatment once the quota recovers, as E-01 did.
  No refusal occurred in the smoke. Detection matches the CLI failure text and archived output
  against: rate limit, usage limit, too many requests, quota, "limit reached", "try again at/in"
  (case-insensitive; `QUOTA` in `launch.py`, unit-tested with simulated refusals). A probe call
  every 30 minutes decides when the quota has recovered; at most 6 set-asides per root, after
  which the root stays missing.
- No dollar cap applies (subscription); the 56-unit work ceiling and candidate safety ceiling are
  the budget. Whether E's $0.75 per-root cap ever bound an E LLM arm is checked in the analysis.

**Everything else is Study E's, byte-for-byte.** The same 90 rows of E's `SEEDS.json`, E's
initialization archives (`initialization/`), task files, campaign plans, ladder, islands,
feedback profile, proposal modes, retry and grammar-aware-repair rules, 56-unit work ceiling,
candidate safety ceiling, search selection rule and multi-seed protocol
(`evaluation_replicates: 4`, search seeds as in E). The registration directory carries an
`INHERITANCE.json` hashing every inherited file against E's registration. Each Sol arm is paired
with E's GP arm by **re-certifying the paired initialization barrier exactly as E's amendment
E-01 did** (`experiments/study_e_20261005/P2_AMENDMENT_01.json`; initialization is deterministic
and makes no API calls).

**Arms.** Sol: new in F. GP: E's archives, read-only. DeepSeek: E's LLM archives, read-only; for
the 12 E-01 roots, the rerun arm that E analysed.

**Validation re-selection (new; identical for all three arms).** Motivation, disclosed: in E a
search-seed advantage held on fresh seeds only about two-thirds of the time (exploratory
cross-tab: when the LLM champion was better on search seeds it won 20 of 30 decisive final
comparisons; when GP's was better, the LLM lost 29 of 36). Validation reduces selection noise
and applies symmetrically to every arm. Procedure, per root and arm:
1. Candidates: the arm's top 5 distinct programs by search fitness (`CandidateFitness.sort_key`
   over completed L2 records including initialization records, proposal-sequence tie-break; the
   same pool as E's selection rule). The first is the arm's search champion.
2. Each candidate is scored on a validation grid: 8 seeds per scenario, all three bundles, base
   constraints; seeds `derive_seed(root_seed, "study-f-validation", scenario_id, j)`, j = 0..7,
   checked disjoint from every search, final, transfer and zero-shot seed in `SEEDS.json`.
3. The validated champion is the candidate with the best validation `fitness_sort_key` (an
   invalid program ranks last); ties are broken by search `sort_key`, then proposal sequence.
k = 5 and 8 seeds were chosen to fit a one-day compute budget (about 90 s per evaluation). The
search champion, top-5 list and validated champion of every arm are sealed before any final grid.

**Final evaluation.** Labels `sol_champion` (Sol search champion), `sol_validated`,
`gp_validated` and `deepseek_validated` are scored on the root's E final grid (eight final seeds
per scenario, all three bundles, base constraints). These final seeds were already used by E to
score E's labels, but never for selection by anyone, in E or F. If E already scored a program
with the same source hash on that grid, E's record is copied and the copy is recorded (scoring is
deterministic: Study D scored 273 programs twice with 0 mismatches). There is no zero-shot arm
and no transfer evaluation.

**Endpoints (fixed sequence).** Both use the feasibility-first directions of the frozen
statistics module (sha256 `b42bd4fd…`, tolerance 1e-9, same file as E), pooled over the 90 roots,
with an exact two-sided sign test at α = 0.05; a directional claim requires p ≤ 0.05 and more
Sol wins than losses.
- **S1 (first):** `sol_validated` versus `deepseek_validated`: the engine effect under identical
  selection.
- **P (second):** `sol_validated` versus `gp_validated`, tested at α = 0.05 only if S1 supports
  Sol. Otherwise P is descriptive only.

The order was fixed before any F data, on expected power: Sol had room to improve on E's
engine, whose champion was its initialization best in 43 of 90 roots.

**Descriptive (unadjusted).** The E-style comparison: `sol_champion` versus E's GP search
champion (E's primary with the engine swapped); validated versus search champion for each arm
(does validation help?); per-family (T/W/R) S1 and P; each validated champion versus the root's
oracle-best reference (E's definition); final-grid feasibility and mean energy when feasible;
jointly feasible energy savings with bootstrap intervals (E's `analysis` seeds); every endpoint
excluding the 12 E-01 roots; roots where compared champions are the same program; calls, failed
calls, CLI-reported tokens, call latency, per-root wall time and work units; quota set-asides.

**Power, stated plainly (S1 and P alike).** At E's tie rate (24/90, about 27%) roughly 66 roots
are decisive; p < 0.05 two-sided then needs about a 42–24 split (64% of decisive roots; 41–25 is
not enough). E's primary was 27–39. F can detect only a large effect; non-significance will not
establish equivalence.

**Execution.**
- Study E launcher adapted to run only the Sol arm plus validation jobs for all three arms;
  registration directory; roots in E's interleaved index order (T00, W00, R00, T01, …).
- Up to 10 Sol roots at once plus validation jobs, gated by memory back-pressure on the 16 GB
  host (a child starts only with 3 GiB available; per-child RSS watchdog), up to 3 concurrent
  validation/final evaluations; disk-space guard, crash recovery and reconciliation inherited.
  Pilot: smoke Sol calls 10/10 succeeded, latency 59–240 s at high effort, no tool calls, no
  refusals; E's search evaluations take ~50 s (peak RSS ~0.3 GiB), validation/final
  evaluations ~12 s per seed per scenario. Projected: ~52 calls per root at ~55 s plus ~36 min
  of evaluation, ~1.6 h per root, ~15–18 h total with 10 roots at once if memory allows.
- A root whose arm cannot complete stays in the ledger as missing; missing primary observations
  block confirmatory inference. No journal editing, no replacement roots. Selections are sealed
  before any final grid; outcomes are sealed before analysis, which uses a frozen `analyze.py`.
- **Smoke before freeze.** E's three smoke roots (ST00, SW00, SR00; reserved seeds, never
  scientific roots), each Sol arm paired to E's smoke GP arm, with smoke validation of the top 2
  candidates on 1 seed per scenario derived with `"study-f-smoke-validation"`. The smoke checks
  the engine wrapper and its pins (model, CLI series, reasoning effort), isolation (empty
  working directory, read-only sandbox, no tool calls observed), the pair certificate against
  E's GP arm, validation of all three arms, the sealing order and final scoring with reuse of E's
  records. The quota path (refusal detection, set-aside, rerun) is covered by a unit test with a
  simulated refusal, not by a live refusal; crash recovery is inherited unchanged from E's
  launcher (tested in E's smoke v2). Smoke records are committed with the
  freeze; smoke outcomes are not analysed.
