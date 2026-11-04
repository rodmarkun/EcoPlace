# Corrected study A — scoped prospective design, 19 September 2026

This version supersedes `CORRECTED_STUDIES_20260917_v1`. It registers **one** study —
the proposal-operator comparison — over **twelve** paired search roots at the unchanged
56-unit work ceiling. It neither repairs historical registrations nor certifies original
P1/P2, and **execution remains blocked** until the prerequisites in GATES.json are met.

## Why there is a new version

Two reasons, both declared in DEVIATION_FROM_V1.json.

**Money.** v1 registered 60 paired roots across two studies. The 17 September resource
pilots measured $0.0398 per model request and 1.875 requests per work unit including
repairs, so $3.32 per paired root and **$198 for v1's design** — against the researcher's
authorized **$60** total. Twelve study-A roots is the largest full-depth design that fits
the $42.65 usable after prior spend and a 40% variance reserve; thirteen would project
$43.19. Study B is dropped rather than shortened, and the per-root search depth is
preserved rather than traded away, because a shallow search produces weak evidence about
search whatever its sample size.

**Source.** v1's study-B pilot aborted against a 2 GiB per-evaluation memory ceiling
caused by retaining per-episode ledgers that the search never reads. The 18 September
remediation changed registered evaluation source, so the frozen source archive is rebuilt.
Replaying every retained pilot evaluation under the remediation reproduces fitness,
episode digests, certificate seals and rung decisions, and leaves operator feedback
byte-identical, so the change is a resource remediation with demonstrated — not asserted —
scientific neutrality.

**No outcome informed either change.** No root of study A or study B was ever executed
under v1: its freeze records `design-frozen-execution-blocked` with `launch_allowed`
false, and `check-launch` refuses. The only executions are two LLM-only resource pilots
with no GP arm, no paired comparison and no final grids. The reduction is a budget
decision taken before any comparative observation, which is exactly what v1's own
protocol required: *"If this count is unaffordable, create a dated design version before
any adaptive study outcome; do not silently shrink a running study or call it adequately
powered."* This document is that version, and it does not call itself adequately powered.

## What this version can and cannot claim

**Claim: proposal-operator comparison.** Twelve new paired search roots on the historical
15-node tight task. Both arms start from the same seven original references and five
seeded mutants. One selected champion per arm is scored on eight fresh per-root seeds and
all three development bundles. The claim is conditional on this task, representation,
proposer settings, selection machinery and resource limits — not universal LLM
superiority.

This is the comparison the historical 16–0–4 result could not support, because identical
initial programs received different constraint treatment (80/240 pairs entered different
islands, 46 received different fitness despite identical simulation metrics), parent-update
cadence differed, and three LLM runs stopped far below the planned work ceiling. Matched
initialization, matched work ceiling and matched evaluation are the point of this study.

**Not claimed, and not merely unproven: additional practical value.** v1's study B asked
whether search adds anything after the discovered simple rule is supplied, in
independently designed conditions. **This version drops that question entirely.** It
cannot speak to transfer, to the strong-reference comparison on independent scenarios, or
to whether practitioners benefit from search. The manuscript must not imply otherwise, and
the completed independent calibration must not be presented as if it stood behind an
executed study.

Also not claimed: hardware energy, carbon, online adaptation, fleet scalability, and
PPO/GPHH state-of-the-art superiority.

## Population, seeds and scheduling

The twelve roots are a **verbatim prefix** of v1's frozen study-A population, A00 through
A11, with their root seeds, search grids, final grids and first-arm assignment unchanged.
Six run the LLM arm first and six the GP arm first, so v1's arm-order balance survives the
subset exactly. Nothing is re-drawn: SEEDS.json records the inheritance and the v1 hash it
came from, and lists every dropped root by name. The initialization archives are
regenerated from those root seeds and required to be byte-identical to v1's frozen
archives, which proves both determinism and subset identity.

Execute paired roots in fixed index order, alternating which arm runs first. One
simulation child at a time per root. All final outcome analysis is held until the complete
population and its selection/source records are sealed. No optional extension to reach
significance, no replacement roots, no choice of favorable final seed grid.

## Exact execution treatment

Unchanged from v1 and inherited byte-identically: task, scenarios, horizons and
constraints (`task_A.json`); ladder rungs, promotion thresholds and epsilon-audit policy;
islands, feedback profile and evolution profile; GP operator configuration (`gp.json`);
model, generation settings and reasoning effort (`engine.json`).

Each arm has a ceiling of 56 normalized work units including initialization and epsilon
audits; L0/L1/L2 cost 0.1/0.3/1.0. The candidate safety ceiling is 562 with three operator
attempts per candidate. Report actual work, terminal state and starting/champion identity
for every arm. Whole-candidate stopping may leave work unspent; no equal-realized-work
claim is made when an arm terminates early. A root's own cost ceiling may end its search
before 56 units — that is a recorded outcome of the registered budget, not a retry
condition.

Final champion selection uses `CandidateFitness.sort_key()` over completed full-evaluation
records with recorded proposal sequence as the final tie-break. Selection cannot use
final-grid results.

## Endpoints and inference

Feasibility-first directions, as in v1: feasible beats infeasible; both infeasible tie;
when both are feasible, smaller mean normalized energy wins. Feasibility means all
constraints pass over the complete root's scenario × final-seed × bundle grid. Absolute
energy differences within 1e-9 normalized units are numerical ties.

**There is exactly one primary hypothesis**, so v1's Holm correction over a family of two
no longer applies and `holm_two` is not used. The primary test is a two-sided exact sign
test over non-tied root directions at alpha 0.05. Dropping a test does not license
spending its alpha elsewhere: alpha is unchanged, not reallocated. A significant result in
the opposite direction is reported as a disadvantage. Any missing arm makes the primary
summary incomplete and blocks any confirmatory claim.

**The decision rule this sample size actually implies.** With no ties the LLM arm must win
**at least 10 of 12** roots to reject at 0.05 (p = 0.0386; 9 of 12 gives p = 0.146). Every
tie raises the required share: 10 of 11, 9 of 10, 8 of 9. At **eight or fewer** non-tied
roots only unanimity can reject, and at five or fewer the test cannot reject at all. The
same thresholds apply symmetrically to a GP advantage, which would be reported as a
disadvantage finding. ANALYSIS.json enumerates every case and derives this prose from the
same computation, so the two cannot drift apart. The coarseness is stated here so that
a near-miss is read as the design's resolution limit rather than as weak evidence of an
effect.

**Power, stated against this design rather than around it.** At a 0.8 per-root win
probability with no ties, power is **0.56**; at 0.7 it is 0.25; only at 0.9 does it reach
0.89. v1 at n=30 had 0.94 at 0.8. **This design is powered only for a large effect.** A
non-significant result does not establish equivalence and must not be reported as one; it
will be reported as an underpowered non-rejection with its confidence interval and every
root direction published. The historical confounded 16–0–4 is not used as a reliable
effect-size estimate, though it is worth stating that an effect of that magnitude, were it
real and unconfounded, would be detectable here.

Always publish absolute feasibility counts and discordance, every root direction, terminal
states, and the jointly feasible energy denominator. Report exact 95% Clopper–Pearson
intervals for conditional win probability, clearly unadjusted. Report mean and median
paired normalized-energy difference for jointly feasible roots with a descriptive 95%
percentile bootstrap interval (20,000 PCG64 resamples of independent roots, linear
quantiles); no interval below two jointly feasible roots. Conditional results do not
estimate energy savings for deployments that fail service delivery.

Practical interpretation stays separate: energy improvements must exceed 0.01 normalized
units and feasibility-frequency improvements 0.10 to be planning-relevant. Publish
directions with the 0.01 tolerance without another significance claim. Comparisons against
fixed references, GP-versus-initialization, champion-versus-own-initialization, search
trajectories and costs are descriptive.

## Missingness, failures, stopping and costs

All twelve roots stay in the ledger. A completed but infeasible policy is not missing. A
valid selected program from an early cost or candidate stop is evaluated and retained with
its actual work and failure cause. A run lacking a trustworthy selected source or final
score is missing; do not invent energy or treat it as a both-infeasible tie. Report
sensitivity assigning each missing direction to either extreme, without promoting the most
favorable assignment. All source and integrity failures stop execution for investigation.

Durable transport requires one HTTP attempt and no automatic unresolved-dispatch retry.
Completed responses replay byte-identically; unknown provider outcomes retain their
monetary reservation and stop the campaign. No journal editing, no replacement-root
rescue. A technical amendment may preserve completed source-identical work; scientific
changes require a new version with untouched grids.

MODEL_AND_RESOURCES.json records the measured pilot costs this scope rests on, the $60
authorization, and the projection: **$39.86 API, $55.81 with a 40% variance reserve**.
Engine settings are inherited unchanged — GLM-5.3, temperature 0.7, top_p 0.8, 24,000
maximum output tokens, one HTTP attempt, reasoning `high`, routed to Z.ai without
fallbacks and with parameter support required. Accounting rates stay $1.40/M input and
$4.40/M output without cache discounts; these are accounting assumptions, not an invoice
or joules.

Over 90% of measured request cost is completion tokens and 80–85% of those are reasoning
tokens. Lowering `reasoning_effort` is therefore the largest available cost lever, and it
is **deliberately not taken here**: it is part of the operator under study, so changing it
would change the treatment and would require its own pilot and its own version.

## Capacity

The 18 September retention remediation is what makes this design executable on the
available machine. All fifteen retained study-A pilot evaluations replay at **at most 278
MiB** against a 2048 MiB ceiling, and evaluation cost fell from 20.8 to 17.3 seconds per
work unit. Projected serial compute is recorded in MODEL_AND_RESOURCES.json.

These are single-evaluation measurements. **Concurrent behaviour on this 16 GB machine is
not measured**, and a launcher must gate on available memory rather than assume the
three-root figure. G0-concurrency is listed as pending for that reason.

## Known failing gate

The scorer-latency risk gate fails on this hardware for all four seed policies at
5.265–6.853 ms p99 against a 5 ms budget (2026-09-18 suite). It pre-dates this design and
is unrelated to the retention remediation — `evolution/risk_gates/gates.py` imports
nothing from the changed modules — but any claim this paper makes about measured execution
constraints must address it, either by stating the hardware and re-deriving the budget or
by narrowing the claim. It is listed as `failing`, not pending, and not as a PASS.

## Prerequisites

GATES.json enumerates the required evidence and distinguishes what the pilots now measure
(proposer and rung timings, per-evaluation memory) from what remains outstanding (capacity
and monetary authorization, concurrency, P1-DES, P1-G2, P1-G3-cal, P1-G-ladder, the
scientific launcher, P2). The old P1/P2 guards and the `development-provisional` config
labels remain intact; a design freeze cannot turn them into scientific authorization, and
`--development` is not a workaround. Unavailable prerequisites are listed as blockers,
never marked PASS. Failure inflation is marked `weakly-measured`: one repair in three
adaptive candidates is a thin basis, and the budget must tolerate a worse rate.

## Sources checked

Carried from v1's 17 September checks; no new external source is relied on by this
scope reduction.

- [GLM-5.3](https://openrouter.ai/z-ai/glm-5.3) — model, reasoning and provider information, listed Z.ai prices.
- [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection) — explicit provider routing, parameter support, fallback controls.
- [Exact binomial inference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) — exact binomial test and exact proportion confidence intervals.

The v1 freeze, operating-level receipt, historical task, baseline sources, current runner,
statistics implementation, pilot measurements, tests and generated design inputs are
hash-bound by this version's freeze.
