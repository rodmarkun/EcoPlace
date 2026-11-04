**P1 G-ladder fidelity audit — 24 September 2026**

The corpus, audit seeds and criteria were committed before any evaluation (`c51b447`,
[PLAN.json](artifacts/PLAN.json)). The corpus is a seeded stratified sample of distinct,
compilable historical programs (LLM, GLM, same-DSL GP and other-arm offspring) plus 12 fixed
references, with every v3 initialization source excluded. Each policy ran at L0, L1 and the full
L2 grid regardless of rung decisions, on audit seeds disjoint from every search, final, pilot and
historical seed. No search, model call or selected champion was involved.
Summary: [SUMMARY.json](artifacts/SUMMARY.json).

| | Study A | Study B |
|---|---|---|
| Policies | 132 | 72 |
| L1 reject rate | 20% | 66% |
| L2-feasible | 62 | 17 |
| **F1**: L2-feasible rejected by L0/L1 (≤ 5%) | **0/62, pass** | **0/17, pass** |
| **F2**: L2 top 5 promoted past L1 | **pass** | **pass** |
| F3: Spearman, L1 vs L2 ranks (descriptive) | 0.89 | 0.97 |
| Runtime-trap policies | 0 | 2 |
| **Ladder** | **PASS** | **PASS** |

Two B policies, both GP offspring, trap at runtime (`list index -1 out of bounds`) during L1. The
real ladder marks such candidates invalid. A supplementary check added after the results confirms
that both also trap on the full L2 grid ([TRAP_L2_CHECK.json](artifacts/TRAP_L2_CHECK.json)), so
they cannot be L2-feasible false eliminations. The first draft of the analyzer also required zero
evaluation errors, which the registered rule does not. It was aligned to the registered decision
(F1 if evaluable, and F2) before this report, and the traps are reported separately.

Limits: one audit seed per scenario, and corpus policies drawn from historical searches on the A
task. B's F1 rests on 17 feasible policies (Clopper–Pearson 95% upper bound on the false-elimination
rate ≈ 18%). The L1 screen is therefore not shown to be harmless for rare near-boundary B policies.
It is shown not to remove any feasible or top-ranked policy in this corpus.
