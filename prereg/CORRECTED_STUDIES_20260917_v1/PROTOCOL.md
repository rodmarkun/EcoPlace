# Corrected studies A and B — prospective design, 17 September 2026

This is a new, versioned design following inspection of historical results and the completed
independent-family calibration. It neither repairs historical registrations nor certifies
original P1/P2. The frozen bundle records scientific choices and current executable inputs;
**execution remains blocked until the separately enumerated prerequisites are satisfied**.
No acceptance fixture, final outcome or paid model is evaluated by this preparation.

## Claims and study populations

**A: proposal-operator comparison.** Thirty new paired search roots on the historical
15-node tight task. Both arms start from the same seven original references and five seeded
mutants. One selected champion per arm is scored on eight fresh per-root seeds and all three
development bundles. The claim is conditional on this task, representation, proposer settings,
selection machinery and resource limits, not universal LLM superiority.

**B: additional practical value.** Thirty separately seeded paired roots. Each search evaluates
all four admitted conditions: distributed/edge-heavy × steady/two-bursts, pressure0.15.
Both arms start from seven references, the exact posthoc bottleneck-headroom reference and
four mutants. B's primary contrast is LLM champion versus the common best initialization
member, selected using initialization search data alone. This asks whether additional search
helps after supplying the discovered simple rule. The original root807 rule and all nine fixed
references remain descriptive comparators; no comparator is removed for poor calibration.

Search is joint over the four B scenarios, not a separate search per cell. Final evaluation
uses eight new seeds per scenario/root and all three bundles. A root, not an episode, scenario,
bundle or template, is the independent search replicate. Scenario-specific B results are
descriptive views of the same roots. A and B are never pooled as sixty interchangeable roots.
Pressure0.30 and0.45 calibration outcomes remain visible. Additional stress/scale/sensitivity
experiments belong to a separate protocol and are not covertly added to this study.

## Exact execution treatment

Generated task, campaign, engine, baseline, archive and seed files are authoritative. Both arms
use sequential single proposals, the paired initialization receipt, identical constraint-island
routing and parent-update opportunities. Initialization uses no model calls. The current GP
mix remains crossover0.45/subtree0.35/numeric0.20; no hindsight GP-cell selection occurs.
The historical deep-search-feedback profile is retained, including its common tournament,
novelty/repair settings and declared operator-specific feedback. The estimand concerns this
configured proposal operator, including its repair/prompt behavior, not an abstract model-only
effect. Mode probabilities and all implicit defaults are bound by source/config hashes.

Each arm has a ceiling of56 normalized work units, including initialization and epsilon audits;
L0/L1/L2 cost0.1/0.3/1.0. These are evaluator-call weights, not equal seconds across A and B.
The candidate safety ceiling is562, three operator attempts per candidate. Both retain the
same population/island settings. Report actual work, terminal state and starting/champion
identity for every arm. Whole-candidate stopping may leave work unspent. No equal-realized-work
claim is made when an arm terminates early.

A keeps its historical ladder shapes. B uses full900-second episodes, all four scenarios at
L1 with dev-1, and all four × three bundles at L2; L0 remains validity-only. B uses the same
primitive-shortfall threshold1.0 and epsilon-audit fraction0.1 as A. B ladder suitability is
unverified and needs prospective pilot/acceptance evidence; no ranking-fidelity PASS is implied.

Final champion and initialization best use `CandidateFitness.sort_key()` over completed
full-evaluation records, with recorded proposal sequence as the final tie-break. Selection
cannot use final-grid results. Both initial-best source identities must match after parity
verification. The final evaluator applies the base task constraints, not an island-specific
variant. Source-identical labels may share an evaluation only when scenario, seed, bundle,
task and environment also match; labels and statistical denominators remain intact.

## Seeds and scheduling

SEEDS.json explicitly materializes all30A and30B roots, search and final scenario grids,
separate resource-pilot roots, and analysis seeds. Search seeds reproduce the actual runner's
`derive_seed(root, 'sim', 'scenario', id, 'replicate', index)` call. Operator RNGs retain separate
arm namespaces; environment seeds are shared within a pair. Initial archive mutant draws are
recorded, including rejected attempts. All new enumerated seed roles are collision-checked
against each other and the named historical/calibration/acceptance-metadata inventory.
This is a declared inventory, not proof about every historical random draw.

Execute paired roots in fixed index order, alternating which arm runs first; A then B for each
index. One simulation child at a time. All final outcome analysis is held until both complete
study populations and selection/source records are sealed. No optional extension to reach
significance, replacement roots or choice of favorable final seed grid is allowed.

## Endpoints and inference

Both primary contrasts use feasibility-first directions: feasible beats infeasible; both
infeasible tie; when both are feasible, smaller mean normalized energy wins. Feasibility means
all constraints pass over the complete root's scenario × final-seed × bundle grid. Energy is
the equal-weight mean of per-instance energy/idle reference, defined only for feasible policies.
Absolute energy differences within1e-9 normalized units are numerical ties.

There are exactly two primary hypotheses: A LLM-versus-GP and B LLM-versus-initialization.
Use exact two-sided sign tests over non-tied root directions, with Holm correction across two
tests at familywise alpha0.05. All ties give p=1. For a directional benefit, adjusted p≤0.05
and more LLM wins than losses are both necessary. A significant result in the opposite
direction is reported as a disadvantage. Missing observations block the primary family.

Always publish absolute feasibility counts/discordance, every root direction, terminal states,
and the jointly feasible energy denominator. Report exact95% Clopper–Pearson intervals for
conditional win probability, clearly unadjusted descriptive intervals. Report mean/median
paired normalized-energy difference for jointly feasible roots, plus a descriptive95%
percentile bootstrap interval (20,000PCG64 resamples of independent roots; linear quantiles).
No bootstrap interval with fewer than two jointly feasible roots. Conditional results do not
estimate energy savings for deployments that fail service delivery.

Practical interpretation is separate: energy improvements must exceed0.01 normalized units;
feasibility frequency improvements of at least0.10 are the planning relevance threshold.
Publish directions with the0.01 tolerance, without another significance claim. Meeting a
statistical threshold alone is insufficient to claim practical deployment value. Comparisons
against every fixed reference, B LLM-versus-GP, GP-versus-initialization, per-cell effects,
search trajectories and costs are descriptive. No equivalence claim follows from a null test.

Sample size is fixed at30roots per study before comparative observations. POWER.json records
exact illustrative sign-test power at alpha0.05 and0.025 under specified tie/win probabilities;
the latter is the conservative first Holm threshold. Historical confounded effects are not
used as a reliable effect-size estimate. Resource feasibility is separately unestablished.
If this count is unaffordable, create a dated design version before any adaptive study outcome;
do not silently shrink a running study or call it adequately powered.

## Missingness, failures, stopping and costs

All60roots stay in the ledger. A completed but infeasible policy is not missing. A valid
selected program from an early cost/candidate stop is evaluated and retained, with actual work
and failure cause. A run lacking a trustworthy selected source or final score is missing;
do not invent energy or treat it as a both-infeasible tie. Incomplete primary families receive
no confirmatory p-value; report completion/feasibility bounds and observed results descriptively.
Report sensitivity assigning each missing direction to either extreme, without promoting the
most favorable assignment. All source/integrity failures stop execution for investigation.

Durable transport requires one HTTP attempt and no automatic unresolved-dispatch retry.
Completed responses replay byte-identically; unknown provider outcomes retain their monetary
reservation and stop the campaign. No journal editing or replacement-root rescue. An explicit
technical amendment may preserve completed source-identical work; scientific changes require
new untouched grids and must retain the original study's status.

MODEL_AND_RESOURCES.json separates a proposed monetary envelope from researcher authorization
and from measured G0 capacity. The proposed allocation is60×$2.50 plus3×$2.50 for resource
pilots=$157.50, with a$225overall envelope giving30% unallocated reserve. Reserve is not
automatically spendable. A per-run monetary ceiling does not guarantee56work units complete.
Frozen engine settings use the historical GLM-5.3 model, temperature0.7, top_p0.8,24,000maximum
output tokens and one HTTP attempt. Reasoning is explicitly `high`, rather than assuming the
historical numeric6000request has supported semantics. Routing is restricted to Z.ai without
fallbacks and with parameter support required. Model/provider availability, accepted generation
settings, actual prompt bounds and conservative reservation coverage require a resource pilot.
Any necessary change creates a new pre-outcome design version.

Pinned accounting rates are$1.40/M input and$4.40/M output, without cache discounts. These are
accounting assumptions, not a verified invoice or joules. The current provider listing supports
these Z.ai rates. Live model weights/serving behavior are not attested; response replay is the
reproducibility claim. One HTTP attempt does not prove provider-side exactly-once execution.

Child memory/time guards, whole-campaign process-family accounting, total CPU/storage/wall
ceilings and their30% reserves must be verified in G0 before launch. The calibration's
single-episode timings inform a planning table only; they do not measure adaptive rung costs,
multi-scenario peak memory or live model latency. No machine capacity is invented as certified.

## Prerequisites and scope revision

GATES.json enumerates required evidence. The old P1/P2 guards and current `development-provisional`
config labels remain intact. A separate design freeze cannot turn these labels into scientific
authorization. In particular the enhanced profile and commercial engine require an explicit
prospective registration/launcher implementation, not `--development` as a workaround.

The commercial proposer gate must test exact initialization parity, GP deterministic replay,
LLM archived-response replay, successful bounded live generation and uncertainty/cost handling;
it must not require or claim deterministic fresh remote responses. This is proposed new
acceptance wording, not a PASS of the unchanged historical G-ablate gate. DES acceptance,
workload/calibration adequacy, ladder validity and the claim-relevant scheduler surface remain
required or need an explicit permitted narrowing before execution. Hardware energy, carbon,
online adaptation, broad fleet scalability and PPO/GPHH state-of-the-art superiority are not
claims of A/B. Completed calibration is independent synthetic design, not empirical validation.

Before execution, the claim/gate mapping and local source archive must have a reviewable clean
release identity, source-bound gate receipts, measured capacity and an actual authorization.
Any independent public timestamp requirement remains separate; local hashes are not proof of
prior public registration. Unavailable prerequisites are listed as blockers, never marked PASS.

## Sources checked on17September2026

- [GLM-5.3](https://openrouter.ai/z-ai/glm-5.3) — model/reasoning/provider information and listed Z.ai prices.
- [Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection) — explicit provider routing,
  parameter support and fallback controls.
- [Exact binomial inference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html) — exact binomial
  test and exact proportion confidence intervals. The installed SciPy version is source-snapshot metadata.

The local operating-level receipt, historical task, baseline sources, current runner,
statistics implementation, tests and generated design inputs are hash-bound by the freeze.
