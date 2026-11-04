**Corrected A+B study: registered results, 2 October 2026**

Analysis: [`analyze.py`](../analyze.py), applying the frozen statistics module (sha256 `b42bd4fd…`,
matching v3 ANALYSIS.json) to the outcomes sealed in commit `e39c387`. Full output:
[PRIMARY.json](PRIMARY.json).
Population: 30 + 30 paired roots. All selections are sealed and all finals are complete; there
are no missing observations and no final-grid faults. API spend: $31.39 (ceiling $60).

## Primary family (feasibility-first directions, exact two-sided sign test, Holm over two)

| | A: LLM champion vs GP champion | B: LLM champion vs initialization best |
|---|---|---|
| Wins / losses / ties | 8 / 2 / 20 | 2 / 5 / 23 |
| p two-sided / Holm | 0.109 / **0.219** | 0.453 / **0.453** |
| Directional claim (Holm ≤ .05 and more wins) | **No** | **No** |
| Conditional win probability, exact 95% CI | 0.80 [0.44, 0.97] | 0.29 [0.04, 0.71] |
| Feasible on final grid (left / right) | 11 / 9 of 30 | 15 / 20 of 30 |
| Jointly feasible roots | 7 | 15 |
| Energy difference (right − left), median; mean [bootstrap 95%] | 0.006; 0.053 [0.002, 0.110] | 0.000; 0.0002 [0.000, 0.0005] |
| Directions at the 0.01 practical-energy threshold | 6 / 2 / 22 | 0 / 5 / 25 |

Neither primary hypothesis is supported, and no significant result runs in the opposite
direction. A non-significant result does not establish equivalence. In study A, the LLM
champion and GP champion are the same program in 20 of 30 roots. In study B, the LLM champion
is the initialization best in 22 of 30 roots.

## Descriptive (unadjusted, not confirmatory)

**Feasibility on fresh final seeds** (fraction of roots):

| Label | A | B |
|---|---|---|
| latency_greedy | 1.00 | 0.97 |
| network_overhead | 1.00 | 0.67 |
| graph_headroom | 1.00 | 0.00 |
| LLM champion | 0.37 | 0.50 |
| GP champion | 0.30 | 0.63 |
| Initialization best | 0.23 | 0.67 |
| cheapest_tier_bottleneck_headroom (B only) | — | 0.00 |

**Energy when feasible** (mean normalized energy):

| Label | A | B |
|---|---|---|
| LLM champion | 0.660 | 0.489 |
| GP champion | 0.730 | 0.489 |
| Initialization best | 0.754 | 0.490 |
| latency_greedy | 0.834 | 0.490 |
| graph_headroom | 1.269 | — |

- In A, the searched champions use about 20% less modeled energy than the robust references
  when they are feasible. But they are feasible on fresh seeds in only 30–37% of roots, against
  100% for three simple references. Under feasibility-first ordering, latency_greedy beats the
  LLM champion 19–9 (2 ties) and the GP champion 21–5.
- LLM champion vs initialization best in A: 8–0 (22 ties, unadjusted p = 0.008). GP champion vs
  initialization best: 2–0.
- In B, energy among feasible policies is essentially flat (about 0.49), so additional search
  has no energy headroom to exploit. LLM vs GP in B is 3–7 (unadjusted p = 0.34).
- The bottleneck-headroom rule that solved every historical grid is infeasible in all 30 B roots.

## Interpretation notes for the paper

- Selection uses one search seed per scenario; finals use eight fresh seeds. The low
  final-grid feasibility of searched champions shows that selection under this registered
  budget generalizes poorly across arrival seeds. This is a property of the frozen design and
  is reported as a finding, not repaired after the fact.
- Execution deviations are disclosed in P2 amendments 01–04: 12 LLM arms were truncated by an
  outage or reboot and then reconciled and resumed; 11 sealed selections were superseded before
  any final evaluation; and the evaluation, memory and connectivity guards were changed.
