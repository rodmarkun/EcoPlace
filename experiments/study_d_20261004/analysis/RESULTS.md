**Study D (C's roots, one search seed): registered results, 4 October 2026**

Protocol: [prereg/STUDY_D_SINGLESEED_20261004](../../../prereg/STUDY_D_SINGLESEED_20261004/PROTOCOL.md),
frozen before any search (commit `c69252f`). Outcomes sealed in commit `1bf9140` before this
analysis, which applies the frozen `analyze.py` and statistics module (sha256 `b42bd4fd…`).

**Execution.** All 30 roots and 60 arm searches completed (exit 0); no root stopped, no resource
guard hit. A disk-full host fault at 08:19 UTC interrupted roots D00–D05; they were resumed with
the unchanged launcher ([EXECUTION_NOTES.md](../EXECUTION_NOTES.md)). Spend $14.50 (project total
$58.37 of $75). Determinism check: 273 programs scored in both C and D, all nine references
present everywhere, **0 mismatches**.

## Primary (registered, Holm over two)

| | P1: LLM_C vs LLM_D | P2: interaction (C advantage − D advantage) |
|---|---|---|
| Positive / negative / zero | 14 / 10 / 6 | 8 / 6 / 16 |
| Exact two-sided p | 0.54 | 0.79 |
| Holm-adjusted p | 1.00 | 1.00 |
| Directional claim | **not supported** | **not supported** |

Neither hypothesis is supported. Non-significance does not establish equivalence; P2 had low
power by design (PROTOCOL.md).

**P1 decomposition (descriptive).** C's LLM champion was feasible and D's was not in 9 roots (all
9 favour C); the reverse never happened; both infeasible in 5 (ties). In the 16 jointly feasible
roots, D's champion used less energy in 10 and C's in 5 (mean energy difference C − D +0.051,
descriptive bootstrap 95% [0.002, 0.099]). The four-seed objective bought feasibility at an
energy cost; feasibility-first ordering nets these to a null.

## Descriptive (unadjusted)

| Label | C feasible | C energy | D feasible | D energy |
|---|---|---|---|---|
| LLM champion | 0.83 | 0.741 | 0.53 | 0.680 |
| GP champion | 0.73 | 0.791 | 0.33 | 0.740 |
| Initialization best | 0.77 | 0.809 | 0.30 | 0.750 |
| latency_greedy | 1.00 | 0.834 | 1.00 | 0.834 |

| Comparison | W–L–T | p | Mean saving, jointly feasible (n) |
|---|---|---|---|
| LLM_D vs GP_D | **11–1–18** | 0.006 | 4.5% (10) |
| LLM_C vs GP_C (Study C primary) | 13–1–16 | 0.0018 | — |
| GP_C vs GP_D | 12–1–17 | 0.003 | −2.9% (10) |
| Init. best C vs D | 14–1–15 | — | — |
| LLM_D vs initialization best D | 11–0–19 | 0.001 | 4.9% (9) |
| GP_D vs initialization best D | 2–0–28 | 0.50 | 0.1% (9) |
| LLM_D vs latency_greedy | 14–14–2 | 1.00 | 18.5% (16) |
| GP_D vs latency_greedy | 5–20–5 | 0.004 | 11.3% (10) |

Excluding the interrupted roots D00–D05: P1 12–7–5, LLM_D vs GP_D 9–0–15, GP_C vs GP_D 10–0–14.
LLM_D and GP_D were the same program in 16 roots; LLM_C and LLM_D in 6.

## Reading

- **The LLM-over-GP advantage did not depend on the search objective on these roots.** It
  appeared with four seeds (13–1) and with one seed (11–1); the interaction was null. The
  description "robust evaluation unlocks the LLM advantage" is not supported.
- **The four-seed objective raised final-grid feasibility for every searched label** (LLM
  0.53→0.83, GP 0.33→0.73, initialization best 0.30→0.77), at some energy cost. It is a robustness
  effect common to both operators, not an LLM-specific one.
- Study A's null (8–2–20, registered) and D's 11–1–18 point the same way on different roots;
  A's contrast may have been underpowered, but that reading is post hoc.
- Qualifications (PROTOCOL.md): fixed C→D order, unpinned provider revision, no provider seed,
  D designed after C's outcomes.
