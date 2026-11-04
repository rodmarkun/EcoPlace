**Study E (multi-family confirmatory): registered results, 7 October 2026**

Protocol: [prereg/STUDY_E_MULTIFAMILY_20261005](../../../prereg/STUDY_E_MULTIFAMILY_20261005/PROTOCOL.md)
(registration `d563151`, launch freeze `6b68003`, amendment E-01 `daa29da`). Outcomes sealed in
`275d429` before this analysis (frozen `analyze.py`, statistics module sha256 `b42bd4fd…`).

**Execution.** All 90 roots completed: 90 selections, 90 final grids, 30 transfer grids, 90
zero-shot seals; no root stopped; no live arm met an HTTP 403. Amendment E-01: an OpenRouter
workspace monthly budget ($60) refused requests from 13:26 UTC on 6 October; the LLM arms of 12
roots (R23–R26, T24–T27, W24–W27) and 10 zero-shot stages were set aside before any final grid and
rerun from scratch, GP arms reused and pairs re-certified. Project spend $110.67 (about $9.45 of it
conservative charges for refused or failed requests never billed).

## Primary (registered)

| | P: LLM champion vs GP champion (90 roots) |
|---|---|
| Wins / losses / ties | **27 / 39 / 24** |
| Exact two-sided p | **0.18** |
| Directional claim | **not supported** (direction favours GP) |

Excluding the 12 rerun roots: 24–33–21 (p = 0.29). Because P did not support the LLM, the gatekept
key secondaries are **descriptive only**: K1 LLM vs oracle-best reference 28–31–31 (p = 0.79);
K2 LLM champion after N slots vs zero-shot 15–27–48 (p = 0.088, favouring zero-shot).

## By family (descriptive, unadjusted)

| | T (historical) | W (wide) | R (drift) |
|---|---|---|---|
| LLM vs GP | **9–3–18** (0.15) | 9–20–1 (0.061) | 9–16–5 (0.23) |
| LLM vs oracle-best reference | 9–12–9 | 9–15–6 | 10–4–16 |
| GP vs oracle-best reference | 1–11–18 (0.006) | **19–7–4 (0.029)** | **18–5–7 (0.011)** |
| LLM atN vs zero-shot | 1–10–19 (0.012) | 12–9–9 | 2–8–20 |
| LLM vs `latency_greedy` | 11–10–9 | 21–0–9 | 28–0–2 |
| Feasible: LLM / GP / init / zero-shot / oracle ref | 0.60 / 0.63 / 0.63 / 0.67 / 1.00 | 0.70 / 0.90 / 1.00 / 0.80 / 1.00 | 0.93 / 0.93 / 1.00 / 1.00 / 1.00 |

Pooled feasibility: LLM 0.74, GP 0.82, initialization best 0.88, zero-shot 0.82, LLM atN 0.79,
oracle-best reference 1.00. Full LLM vs zero-shot: 24–31–35 (p = 0.42). Transfer of T labels to
drift: LLM champion feasible in 17% of roots, GP 0%, oracle reference 100%.

## Reading

- **H1 not supported.** Across the three task variants the LLM did not beat GP; the observed
  direction favoured GP (27–39). On the historical task alone the LLM again led GP (9–3–18), as in
  studies A, C and D, but on the wide and drift variants GP led (20–9 and 16–9, descriptive).
- **Selection still overfits on the new variants.** The initialization best was more often
  feasible on fresh seeds (0.88) than either champion; on W the LLM champion fell to 0.70 against
  1.00 for the initialization best. Four search seeds were not enough to prevent selection of
  fragile programs there, and the LLM was the more affected operator.
- **Zero-shot LLM proposals did as well as or better than iterative LLM evolution** (K2 15–27,
  descriptive; on T 1–10).
- **GP beat the best hand-written reference** on W (19–7) and R (18–5), descriptively, while
  `latency_greedy` failed outright there.
- Non-significance does not establish equivalence; families are related variants (PROTOCOL.md).
