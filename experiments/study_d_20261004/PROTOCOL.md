**Study D: single-seed counterpart of Study C — internally frozen protocol, 4 October 2026**

**Motivation, disclosed.** Study C (results sealed and analysed on 4 October 2026) found that,
with four arrival seeds per scenario in every full search evaluation, the LLM champion beat the
GP champion on fresh seeds (13–1–16, p = 0.0018). Study A had used one search seed per scenario
and found no registered advantage, but A and C used different roots, so the role of the search
objective was supported only descriptively. Study D is the within-root test. It reruns Study C's
30 roots with Study C's single registered change reversed. Study D is designed after Study C's
outcomes were known, and is reported as a follow-up; it does not alter the A, B or C results.

**Hypotheses.** (H1) On the same roots and the same fresh final seeds, the LLM champion found
with the four-seed search objective (Study C) is better than the LLM champion found with the
one-seed objective (Study D). (H2) The LLM-versus-GP advantage is larger with the four-seed
objective than with the one-seed objective.

**Population and inputs.** The 30 Study C roots, renamed D00–D29 (D*k* pairs with C*k*). Root
seeds, initialization archives, final seeds, task, ladder, islands, feedback profile, GP
configuration and LLM engine are byte-identical copies of the frozen Study C registration
(INHERITANCE.json). Campaign identifiers do not enter any seed derivation.

**The single registered change (relative to C).** `evaluation_replicates` goes from 4 to 1, the
study-A setting. The one search seed per scenario is `derive_seed(root, 'sim', 'scenario', id,
'replicate', 0)`, which is Study C's first search seed. Work weights, ceilings (56 units), the
candidate safety ceiling, the selection rule and the paired initialization barrier are unchanged.

**Final evaluation.** Labels: the LLM champion, the GP champion, the common initialization best
and the nine runtime/policies references, each scored on Study C's eight final seeds per
scenario across all three bundles under the base constraints. Selections are sealed before any
final grid is run. Study C's sealed final records are used unchanged for the C side of each
contrast. Final evaluation is deterministic: a program scored in both studies must produce the
same record. Every program scored in both studies is checked, and all nine references must be
present and completed in both studies for every root; any mismatch or missing reference blocks
confirmatory inference. The initialization best is common to both arms within a study, but it
can differ between C and D because the barrier selects it under the study's search objective;
that difference is part of the treatment.

**Primary endpoints and tests (two, Holm-corrected).** Per root, `dir(x, y)` is the
feasibility-first direction of the frozen statistics module (sha256 `b42bd4fd…`, tolerance
1e-9).
- *P1 (H1):* `dir(LLM_C, LLM_D)`.
- *P2 (H2, interaction):* `sign(dir(LLM_C, GP_C) − dir(LLM_D, GP_D))`.

Each is an exact two-sided sign test over non-zero roots. The two p-values are Holm-adjusted with
the frozen `holm_two`. A directional claim requires a Holm-adjusted p ≤ 0.05 and more positive
than negative roots. Any missing observation makes the family incomplete and blocks confirmatory
claims. A non-significant result does not establish equivalence or absence of an effect.

**Power, disclosed.** P1 is expected to be well powered: the LLM champion was feasible on 0.83 of
roots in C and on 0.37 in A. P2 is a contrast of two ternary directions, and many roots are
expected to tie. A simulation under the study-C and study-A direction frequencies gives P2 a
power of about 0.2 to 0.5, depending on how strongly a root's C and D outcomes are correlated.
A null P2 is therefore weak evidence.

**Secondary, descriptive.** `dir(LLM_D, GP_D)` (a within-root replication of study A's null on
C's roots); `dir(GP_C, GP_D)`; each D champion versus the initialization best and
`latency_greedy`; final-grid feasibility and mean energy when feasible; bootstrap intervals for
jointly feasible energy differences (seed in SEEDS.json `analysis.D`); spend and requests.

**Execution.** The Study C launcher, with registration directory, study letter and prior
exposure changed. Settled no-response and HTTP errors, connectivity and peak-window waits,
memory back-pressure, a $0.95 per-root ceiling and six concurrent pair slots as in C. The
project API ceiling is raised from $60 to $75 by the investigator, so the global ceiling
cannot bind (prior spend plus 30 × $0.95).

**Threats to causal attribution, disclosed.** (i) C and D run in a fixed order, about one day
apart; the provider model is named but its revision and weights are not pinned, so provider drift
is confounded with the objective. (ii) The transport sends no provider seed, so matching local
seeds does not reproduce LLM draws: D is a paired rerun, not a replay, and per-root LLM outcomes
differ by sampling as well as by objective. (iii) D was designed after C's outcomes were known.
D therefore tests the objective within roots under these qualifications, not unconditionally. All 30 roots stay in
the ledger; no journal editing and no replacement roots.
