**Study C (robust objective): registered results, 4 October 2026**

Protocol: [prereg/STUDY_C_ROBUST_20261003](../../../prereg/STUDY_C_ROBUST_20261003/PROTOCOL.md),
frozen before any search. Launch freeze: commit `35cb68d`. Outcomes were sealed in commit
`8deb4fe` before this analysis, which applies the frozen statistics module (sha256 `b42bd4fd…`).

**Execution.** 30 paired roots on study A's task, with search evaluated on 4 arrival seeds per
scenario. All 60 arm searches completed normally; there were no failures, missing roots or
resumed arms. 1,795 LLM requests (1,791 completed, 4 empty) cost $12.44. Wall time was about
8 hours, with 6 pair slots and concurrent arms.

## Primary (registered): LLM champion vs GP champion

| | |
|---|---|
| Wins / losses / ties | **13 / 1 / 16** |
| Exact two-sided sign test | **p = 0.0018** (α = 0.05; single primary test) |
| Directional claim | **supported** (p ≤ .05 and more LLM wins) |
| Conditional win probability (exact 95% CI) | 0.93 [0.66, 1.00] |
| Feasible on final grid | LLM 25/30, GP 22/30 |
| Jointly feasible roots | 22 |
| Energy difference GP − LLM: median; mean [bootstrap 95%] | 0.000; 0.044 [0.017, 0.075] |
| Directions at the 0.01 practical-energy threshold | 10 / 1 / 19 |

The champions were the same program in 15 of 30 roots.

## Descriptive (unadjusted)

**Final-grid feasibility (fraction of roots) and mean normalized energy when feasible:**

| Label | Study C (4-seed search) feasible | energy | Study A (1-seed search) feasible | energy |
|---|---|---|---|---|
| LLM champion | 0.83 | 0.741 | 0.37 | 0.660 |
| GP champion | 0.73 | 0.791 | 0.30 | 0.730 |
| Initialization best | 0.77 | 0.809 | 0.23 | 0.754 |
| latency_greedy | 1.00 | 0.834 | 1.00 | 0.834 |

The A and C columns use different roots and seeds, so they are descriptive context only.

**Pairwise comparisons:**

| Comparison | W–L–T | p (unadjusted) | Mean saving on jointly feasible (n) |
|---|---|---|---|
| LLM champion vs initialization best | 13–0–17 | 0.0002 | 7.1% (23) |
| GP champion vs initialization best | 4–1–25 | 0.38 | 2.1% (22) |
| LLM champion vs latency_greedy | 15–5–10 | 0.041 | 11.1% (25) |
| GP champion vs latency_greedy | 6–8–16 | 0.79 | 5.2% (22) |

## Interpretation and limits

- **Reading.** Under a robust search objective, the LLM proposal operator found programs that
  were significantly better than GP's on fresh seeds. They were more often feasible and, when
  both were feasible, lower in energy. They also beat the robust `latency_greedy` rule on
  15 roots against 5.
- **Motivation, disclosed.** Study C's hypothesis came from exploratory analyses of studies A
  and B. It is a separate prospective test on fresh roots and seeds with one registered change
  and one primary test. It does not alter the A and B results.
- **What the comparison does and does not show.** A and C differ in the search objective but
  were run on different roots. That replication improves the LLM's *relative* advantage is
  therefore supported only descriptively, not by a within-root test.
- **Scope.** One synthetic task and modeled energy. The 4-seed objective costs about four times
  as many simulated episodes per full evaluation.
