**Exploratory post-study analyses — 2 October 2026 (NOT confirmatory)**

These analyses explain the registered results in
[`../corrected_ab_20260925/analysis/RESULTS.md`](../corrected_ab_20260925/analysis/RESULTS.md);
they do not change them. Rules were committed before any job ran (`adbb7c5`,
[PLAN.json](artifacts/PLAN.json)). Two runner amendments changed only the worker count and the
per-study job files ([RUNNER_AMENDMENTS.json](artifacts/RUNNER_AMENDMENTS.json)). They were
designed after the registered final outcomes were known, so they can explain those outcomes
but cannot confirm them independently. No API calls were made.

## 1. Execution cost ([../execution_cost_20261002](../execution_cost_20261002/))

Apple M2 Pro (Mac14,9), idle machine (load about 2). Pinned 200-node cluster under the
unchanged 5 ms gate procedure, measured over 34 distinct sources.

| Group | n | scorer p99 ms, median [range] | fuel ×10³ | lines | passes 5 ms |
|---|---|---|---|---|---|
| LLM champions | 19 | 4.68 [2.15, 5.88] | 400 [1, 561] | 85 [7, 142] | 17/19 |
| GP champions | 10 | 4.44 [2.15, 5.93] | 375 [1, 425] | 34 [7, 36] | 9/10 |
| References | 10 | 2.41 [2.12, 4.87] | 13 [1, 402] | 28 [7, 65] | 10/10 |

- **Proposer p99:** at most 2.6 ms for every GP champion and reference. Eight of 19 LLM
  proposers exceed that: 2.8, 7.7, 7.8, 8.4, 9.0, 19.6, 31.0 and 101.5 ms.
- **Wall time** is dominated by marshalling the 200-node state: `first_fit` uses 1,000 fuel units
  but takes 4.3 ms.

## 2. Failure mode and energy-model dependence (from sealed final records)

**Failure mode** ([SHORTFALL.json](SHORTFALL.json), [SLO_MARGIN.json](SLO_MARGIN.json)):
- Every infeasible champion or initialization best, in both studies, violated only SLO
  attainment.
- In study A the median summed shortfall was 0.011 (LLM) and 0.013 (GP); the worst
  single-instance attainment was 0.859 against the 0.88 threshold.
- The worst template's mean attainment had a median of 0.892 for the champions and 0.923 for
  `latency_greedy`.
- Champions met every SLO on 85% and 84% of final instances; `latency_greedy` met them on 100%.

**Per-bundle energy saving vs `latency_greedy`, study A** ([BUNDLE_GAP.json](BUNDLE_GAP.json)),
over jointly feasible roots:

| Champion | dev-1 | dev-2 | dev-3 |
|---|---|---|---|
| LLM | 36.7% | 30.0% | 8.9% |
| GP | 23.9% | 19.1% | 3.8% |

The bundles change fog capacity and power together.

## 3. Workload-model sensitivity, study A ([SENSITIVITY.json](SENSITIVITY.json))

The frozen workload-sensitivity transform, applied on every study-A final grid:

| Setting | Feasible LLM / GP / init / latency_greedy (of 30) | LLM saving vs latency_greedy (W–L–T) | GP saving (W–L–T) |
|---|---|---|---|
| attribution 0.30 | 30 / 30 / 30 / 30 | 19.5% (28–0–2) | 17.3% (26–0–4) |
| observed mean/p95 shape @0.45 | 10 / 7 / 6 / 29 | 20.9% on 10 joint (8–19–3) | 9.6% on 7 joint (3–22–5) |
| attribution 0.60 | 0 / 0 / 0 / 0 | — | — |

For reference, the registered setting (0.45) gave LLM 20.9% (9–19–2) and GP 12.5% (5–21–4).

## 4. Validation-grid re-selection

Each arm's top-5 distinct L2 search candidates were re-scored on a disjoint validation grid. A
used 4 seeds per scenario and B used 2, across all 3 bundles. The validated champion is the
best feasibility-first sort key on that grid; final seeds were never used for selection.

**Study A** ([VALIDATED_SELECTIONS_A.json](artifacts/VALIDATED_SELECTIONS_A.json); 5 new final
tests). The selection changed in 9 of 30 roots per arm.

| | Final feasible | vs latency_greedy W–L–T | saving on jointly feasible (n) |
|---|---|---|---|
| LLM search champion | 0.37 | 9–19–2 | 20.9% (11) |
| LLM validated champion | 0.60 | 9–12–9 | 12.4% (18) |
| GP search champion | 0.30 | 5–21–4 | 12.5% (9) |
| GP validated champion | 0.60 | 8–12–10 | 9.8% (18) |

Validated champion vs search champion: 7–1–22 (LLM) and 9–0–21 (GP).

**Study B** ([VALIDATED_SELECTIONS_B.json](artifacts/VALIDATED_SELECTIONS_B.json); 8 new final
tests). The selection changed in 11 (LLM) and 3 (GP) roots.

| | Final feasible | vs latency_greedy W–L–T | saving on jointly feasible (n) |
|---|---|---|---|
| LLM search champion | 0.50 | 15–14–1 | 0.13% (15) |
| LLM validated champion | 0.67 | 18–10–2 | 0.66% (20) |
| GP search champion | 0.63 | 18–11–1 | 0.16% (19) |
| GP validated champion | 0.70 | 19–8–3 | 0.21% (21) |

Validated champion vs search champion: 7–0–23 (LLM) and 3–0–27 (GP). Energy differences are
under the 0.01 practical threshold.
