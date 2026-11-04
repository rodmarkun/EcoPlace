**Study C: robust-objective follow-up — internally frozen protocol, 3 October 2026**

**Motivation, disclosed.** Studies A and B (registration v3, results sealed on 1 October 2026)
found no registered advantage for LLM over GP search. Exploratory analyses of those results
then showed two things. First, LLM proposals beat the starting archive far more often than GP
proposals did. Second, the selected champions fail fresh arrival seeds by narrow
service-level misses after selection on a single search seed. Study C tests the hypothesis
those analyses suggest. It is a new, prospectively specified study. It does not re-analyse,
replace or rescue studies A and B, and its results are reported alongside theirs.

**Hypothesis.** When the full evaluation used during search scores every candidate on four
arrival seeds per scenario instead of one, the LLM proposal operator finds placement programs
that are better on fresh final seeds than those found by same-representation GP.

**Population and inputs.** Thirty new paired roots, C00–C29, on study A's historical 15-node
task. The task, ladder, islands, feedback profile, GP configuration and LLM engine (DeepSeek
V4.1 Flash, low reasoning, 32,000 output tokens, temperature 0.7, top-p 0.8, DeepSeek-only
routing) are byte-identical copies of the frozen v3 study-A inputs (INHERITANCE.json).
Initialization archives are built by the same procedure as study A's (seven references and
five mutants) from new root seeds.

**The single registered change.** `evaluation_replicates` goes from 1 to 4. Search (L2) seeds
are `derive_seed(root, 'sim', 'scenario', id, 'replicate', j)` for j = 0..3. L1 keeps one seed
and one bundle, as before. Work weights, ceilings (56 units), the candidate safety ceiling,
the selection rule and the paired initialization barrier are unchanged. Each L2 evaluation
therefore runs four times as many simulated episodes at the same work charge.

**Final evaluation.** Labels: the LLM champion, the GP champion, the common initialization
best and the nine runtime/policies references. Each is scored on eight fresh final seeds per
scenario across all three bundles under the base constraints. These seeds are disjoint from
every search, final, validation, audit and pilot seed already used. Selections are sealed
before any final grid is run.

**Primary endpoint and test.** A single primary contrast: LLM champion versus GP champion,
using the feasibility-first directions of the frozen statistics module (sha256 `b42bd4fd…`).
Two-sided exact sign test at alpha 0.05 (one test, so no multiplicity correction). We report
the exact conditional-win interval, jointly feasible energy differences with the registered
bootstrap (seed in SEEDS.json `analysis.C`), and directions at the 0.01 practical-energy
threshold. A directional claim requires p ≤ 0.05 and more LLM wins than losses. A
non-significant result does not establish equivalence.

**Secondary, descriptive.** Each champion versus the initialization best and versus
`latency_greedy`; final-grid feasibility frequencies; SLO margins; comparison with study A's
single-seed results; cost and work.

**Execution.** The launcher behaves as in P2 amendments 01–04: settled no-response and HTTP
errors, connectivity waits, memory back-pressure, a global $60 ceiling and a $1.50 per-root
ceiling. Scheduling changes only, none of which can alter an evaluation result: up to six
pairs run concurrently, and after the pair's initialization certificate both arms search at
the same time. All 30 roots stay in the ledger; missing observations block confirmatory
inference. No journal editing and no replacement roots.
