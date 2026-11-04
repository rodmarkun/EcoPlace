**Study E: exploratory post-study analyses, 7 October 2026 (NOT confirmatory)**

These analyses explain the registered results in
[`../analysis/RESULTS.md`](../analysis/RESULTS.md); they do not change them. **Every analysis
below was chosen after the registered outcomes were known.** None was named in the
[protocol](../../../prereg/STUDY_E_MULTIFAMILY_20261005/PROTOCOL.md), no plan was committed
before it ran, and the p-values are unadjusted and were computed after the data had been seen.
Read them as hypothesis-generating. A future study would have to test them prospectively.

Source: [`explore.py`](explore.py). It only reads sealed files (`analysis/PRIMARY.json`,
`runs/*/SELECTION.json`, `BUDGET.jsonl`, the transport journals, `SEEDS.json`, and the
`PRIMARY.json` of studies A/B, C and D). It runs no simulation and makes no API call, and its peak
RSS is about 250 MB. Directions use the frozen statistics module (sha256 `b42bd4fd…`, checked at
start), and the per-family LLM-vs-GP counts are asserted equal to `PRIMARY.json`. Run it from the
repository root with `.venv/bin/python experiments/study_e_20261005/exploratory_20261007/explore.py`.

"Rerun roots" are the 12 roots whose LLM arm was rerun under amendment E-01: R23–R26, T24–T27
and W24–W27.

## 1. Task-family heterogeneity of the LLM-vs-GP contrast

| LLM champion vs GP champion | W–L–T | sign p | decisive | same program |
|---|---|---|---|---|
| T (`dev-continuum-tight`) | 9–3–18 | 0.15 | 12/30 | 16 |
| W (`dev-continuum-wide`) | 9–20–1 | 0.061 | 29/30 | 0 |
| R (`dev-continuum-drift`) | 9–16–5 | 0.23 | 25/30 | 5 |
| W + R | 18–36–6 | 0.020 | 54/60 | 5 |

**Tests on decisive pairs (exploratory, post hoc):**
- Fisher exact, T vs W+R, table [[9, 3], [18, 36]]: **two-sided p = 0.0109**, sample odds
  ratio 6.0. This confirms the quick figure of p ≈ 0.011.
- Pairwise: T vs W p = 0.016, T vs R p = 0.038, W vs R p = 0.78.
- Freeman–Halton exact test over all three families (3×2): p = 0.035.
- **Excluding the 12 rerun roots:** T 7–3–16, W 8–17–1, R 9–13–4. Then T vs W+R
  [[7, 3], [17, 30]] gives Fisher p = 0.077 and the 3×2 test p = 0.125. The contrast does not
  survive this sensitivity cut at 0.05.

**Observations.**
- The LLM won exactly 9 roots in every family. The heterogeneity comes entirely from GP's wins
  (3 on T against 20 on W and 16 on R).
- On T, 18 roots are ties, and in 16 of them the two arms selected the same program. On W there
  is 1 tie and the two arms never selected the same program. Conditioning on decisive pairs
  therefore compares a small, selected subset on T with nearly every root on W.
- The pattern follows GP's behaviour more than the LLM's. GP almost never improved on its
  starting archive on T, but often did on W and R (section 2).

**Which task the system was developed on (repository evidence).** Every LLM development and
engineering pilot that ran search in the corrected pipeline used `dev-continuum-tight` (T). Some
also used study B's `distributed-*`/`edge_heavy-*` scenarios.
- `experiments/adaptive_budget_pilots_20260917/artifacts/A/task.json`,
  `deepseek_pilots_20260920/artifacts/A/task.json`,
  `deepseek_pilots_bounded_20260923/artifacts/A/task.json` (its PLAN.json search grid lists
  `dev-continuum-tight` for study A and `distributed-*` for B),
  `deepseek_concurrency_20260923/artifacts/A0/task.json` and
  `g_ablate_commercial_20260924/artifacts/task.json`: all `scenario_ids` are
  `dev-continuum-tight`.
- The archive seed's header comment, sent in every proposal prompt (e.g.
  `runs/T00/llm/journal.jsonl`), justifies the seed by T's properties ("worst SLO 0.88 against a
  0.90 target").
- **GP was also developed on T.** `experiments/gp_tuning_20260902/GP_TUNING.json` (operator mix
  and budget) references only `dev-continuum-tight`.
- Before Study E, W and R appear only as evaluation targets with no search on them:
  `experiments/transfer_20260902/` (a rescore of T-searched champions with zero LLM calls) and
  `validation_20260909/fit_diagnostics_*`.
- The history was squashed into commit `82350a07` (24 Sep), so git dates cannot show when each
  scenario file was created. The evidence above comes from the experiment directories.

T is therefore the development task for **both** operators. The T-versus-W+R contrast cannot
separate "LLM tuned to T" from "GP tuned to T", from properties of T itself, or from chance.

## 2. Cross-study view of the historical task (descriptive only)

| Study | Task | Search seeds | LLM vs GP W–L–T | Status of that contrast |
|---|---|---|---|---|
| A | T | 1 | 8–2–20 | registered primary, not significant (p = 0.11; Holm 0.22) |
| C | T | 4 | 13–1–16 | registered primary, significant (p = 0.0018) |
| D | T (C's roots) | 1 | 11–1–18 | descriptive |
| E, family T | T | 4 | 9–3–18 | descriptive per family |
| B | distributed / edge-heavy | 1 | 3–7–20 | descriptive |
| E, family W | W | 4 | 9–20–1 | descriptive per family |
| E, family R | R | 4 | 9–16–5 | descriptive per family |

Sources: `corrected_ab_20260925/analysis/PRIMARY.json` (A primary; B
`llm_champion vs gp_champion`), `study_c_20261003/analysis/PRIMARY.json`,
`study_d_20261004/analysis/PRIMARY.json` (`llm_champion@D vs gp_champion@D`), and Study E
`PRIMARY.json`.

On T, the LLM led in direction in all four studies. Over the three independent root sets
(A + C + E-T; D reuses C's roots) the sum is **30–6–54**. No pooled p-value is given: the studies
differ in protocol, and the cross-study grouping was chosen after seeing the results. The two
non-T tasks with a GP arm, B and E's W/R, both went GP's way in direction.

**How often each operator improved on its own starting archive** (champion vs initialization
best, W–L–T):

| | A (T) | C (T) | D (T) | E-T | E-W | E-R |
|---|---|---|---|---|---|---|
| GP | 2–0–28 | 4–1–25 | 2–0–28 | 1–0–29 | 19–7–4 | 18–4–8 |
| LLM | 8–0–22 | 13–0–17 | 11–0–19 | 9–2–19 | 9–15–6 | 8–3–19 |

On T, GP almost never left its initialization: 9 wins in 120 roots across the four studies. The
LLM improved in 41 of those 120 roots. On W and R the positions reverse: GP improved often, and on
W the LLM champion lost to its own initialization best in 15 roots (see section 4).

## 3. Iteration vs zero-shot, with cost

**Outcomes (W–L–T, sign p).** "atN" is the LLM arm's champion after N proposal slots, where N
is the number of slots the zero-shot arm used.

| | K2: LLM atN vs zero-shot | Full LLM vs zero-shot | atN vs ZS mean saving, jointly feasible (n) |
|---|---|---|---|
| T | 1–10–19 (0.012) | 5–10–15 (0.30) | −16.9% (18) |
| W | 12–9–9 (0.66) | 12–13–5 (1.00) | +2.2% (20) |
| R | 2–8–20 (0.11) | 7–8–15 (1.00) | −11.0% (29) |
| Pooled | 15–27–48 (0.088) | 24–31–35 (0.42) | −8.6% (67) |
| Pooled, excl. rerun roots | 12–24–42 (0.065) | 21–27–30 (0.47) | −7.0% (57) |
| T, excl. rerun roots | 1–8–17 (0.039) | 4–8–14 (0.39) | −15.5% (15) |

A negative saving means the zero-shot champion used less energy. Mean N was 17.5 on T, 17.8 on W
and 17.9 on R slots, out of 20. Zero-shot proposals promoted to L2 averaged 7.1 (T), 10.1 (W) and
8.3 (R) per root.

**Units and reconciliation.** `BUDGET.jsonl` amounts are **micro-USD**.
- The launcher counts the last settlement per request key. On that basis Study E's ledger
  exposure is **$51.97**. Adding the launcher's prior exposure of $58.70
  (`launch.prior_exposure_microusd()`) gives **$110.67**, which matches RESULTS.md.
- The raw sum of every settle event is $54.39. The difference arises because 87 rerun requests
  repeated a superseded request key (requests are deterministic), and the ledger keeps only the
  rerun's settlement for those keys.
- From the journals, the provider-billed cost is **$44.94** for Study E. There were $9.45 of
  conservative $0.05 charges on 189 HTTP-error requests that were never billed. Only $7.35 of
  these remain in the de-duplicated $110.67 figure; the other $2.10 sit on keys whose rerun
  settlement replaced them.

**Spend and tokens per arm.** Ledger and billed amounts are in USD. Tokens come from the journals'
`usage` objects; reasoning tokens are a subset of completion tokens.

| Scope | Arm | Roots | Ledger $ | Billed $ | Conservative $ | Requests | Prompt tok | Completion tok (reasoning) | Ledger $/root |
|---|---|---|---|---|---|---|---|---|---|
| All journals, incl. superseded E-01 | LLM | 90 | 37.12 | 31.35 | 7.20 | 4,926 | 19.02M | 47.50M (41.11M) | 0.412 |
| | Zero-shot | 90 | 14.85 | 13.58 | 2.25 | 2,057 | 6.48M | 21.02M (18.63M) | 0.165 |
| Current journals (rerun arms once) | LLM | 90 | 32.13 | 30.83 | 1.30 | 4,726 | 18.74M | 46.70M (40.40M) | 0.357 |
| | Zero-shot | 90 | 13.89 | 13.44 | 0.45 | 1,997 | 6.39M | 20.79M (18.43M) | 0.154 |
| Current, excl. 12 rerun roots | LLM | 78 | 27.98 | 26.68 | 1.30 | 4,094 | 16.26M | 40.40M (34.96M) | 0.359 |
| | Zero-shot | 78 | 12.05 | 11.60 | 0.45 | 1,727 | 5.53M | 17.95M (15.91M) | 0.154 |

Note on the first scope: its ledger column covers every ledger key, whereas its billed and
conservative columns also count the superseded settlements that the ledger dropped. The
superseded E-01 journals alone hold 260 requests (154 HTTP 403/404), $0.67 billed and $7.70
conservative. E-01 inflates the LLM arm's all-in ledger spend from $32.13 to $37.12.

**Current-journal ledger spend by family, per root:**

| | T | W | R |
|---|---|---|---|
| LLM | $0.431 | $0.278 | $0.361 |
| Zero-shot | $0.154 | $0.152 | $0.156 |

Excluding the rerun roots: LLM $0.436 (T), $0.278 (W), $0.362 (R); zero-shot about $0.15
throughout.

Per root, the full LLM arm cost about 2.3 times the zero-shot arm ($0.357 vs $0.154) and used
about 2.4 times the tokens (0.73M vs 0.30M). The zero-shot arm's outcomes were no worse in
direction.

## 4. Selection overfitting: search fitness vs final-grid feasibility

Search fitness is the selection record in `SELECTION.json`: four search seeds, scored under the
constraints of the island the record was evaluated on. Final feasibility is on eight fresh seeds
under base constraints. Each selection's `source_sha256` was checked against `PRIMARY.json`.

| Family | Label | Search-feasible | Final-feasible | Final-feasible given search-feasible | Search energy → final energy (both feasible) |
|---|---|---|---|---|---|
| T | LLM champion | 30/30 | 18/30 | 18/30 | 0.736 → 0.737 |
| T | GP champion | 30/30 | 19/30 | 19/30 | 0.804 → 0.805 |
| T | Initialization best | 30/30 | 19/30 | 19/30 | 0.814 → 0.814 |
| T | Zero-shot champion | 30/30 | 20/30 | 20/30 | 0.714 → 0.713 |
| W | LLM champion | 20/30 | 21/30 | 13/20 | 1.262 → 1.263 |
| W | GP champion | 18/30 | 27/30 | 16/18 | 1.110 → 1.110 |
| W | Initialization best | **0/30** | **30/30** | — | — |
| W | Zero-shot champion | 10/30 | 24/30 | 8/10 | 1.282 → 1.282 |
| R | LLM champion | 30/30 | 28/30 | 28/30 | 1.164 → 1.165 |
| R | GP champion | 30/30 | 28/30 | 28/30 | 1.219 → 1.221 |
| R | Initialization best | 30/30 | 30/30 | 30/30 | 1.279 → 1.279 |
| R | Zero-shot champion | 30/30 | 30/30 | 30/30 | 1.177 → 1.177 |

LLM atN, for reference: T 19/30 final-feasible, W 23/30 (search-feasible 9/30), R 29/30.

**Observations.**
- **On T**, every selected program was feasible on all four search seeds, yet a third to two
  fifths failed on fresh seeds (18–20 of 30 feasible). Energy is reproduced almost exactly from
  search to final grid. The overfitting is in SLO feasibility, which matches the boundary finding
  in [brittleness_20261002](../../brittleness_20261002/REPORT.md), not in energy.
- **On W, search feasibility does not measure robustness.**
  - W's initialization best is `kube_least_allocated`, which is the oracle-best reference on all
    30 W roots. It was search-infeasible in every root, because it was scored on the stricter
    qos-tight island (PROTOCOL.md), yet it was feasible on every final grid.
  - LLM champions that were search-feasible were final-feasible in 13 of 20 roots. GP's were
    final-feasible in 16 of 18.
  - Read with care: on W the selection rule compares records judged under different island
    constraints.
- **On R**, nearly everything is feasible, and the contrasts are decided by energy.

**Transfer: T selections scored on 8 fresh `dev-continuum-drift` seeds.** The native-R column
shows the same label when searched on R itself.

| Label | Feasible on drift | Native R feasibility |
|---|---|---|
| Oracle reference / `graph_headroom` | 30/30 | 1.00 |
| `least_allocated` | 10/30 | 0.33 |
| Zero-shot champion | **8/30** | 1.00 |
| LLM champion | 5/30 | 0.93 |
| `kube_least_allocated` | 2/30 | 0.07 |
| LLM atN | 1/30 | 0.97 |
| GP champion | 0/30 | 0.93 |
| Initialization best | 0/30 | 1.00 |
| `latency_greedy`, `network_overhead`, `first_fit`, `energy_greedy`, `most_allocated`, `kube_most_allocated` | 0/30 | 0.00 |

The registered drift contrasts were LLM vs GP 5–0–25 (p = 0.062) and LLM vs oracle reference
5–25–0. Champions selected on T rarely survive drift, but searching on R directly yields feasible
champions in 93% of roots. Of the searched labels, the zero-shot champion transferred most often.
A pre-correction development rescore showed the same direction: 6/20 LLM vs 0/20 GP feasible on
drift (development experiment `transfer_20260902`, not included in this repository).

## 5. GP (and LLM) vs hand-written references and initialization

The oracle-best reference is chosen with hindsight per root from nine references. The calibration
reference is the family's reference that was feasible at the pre-freeze headroom gate, so it is
not chosen with hindsight: `latency_greedy` (T), `kube_least_allocated` (W), `graph_headroom` (R).

| | GP vs oracle-best | GP vs initialization best | GP vs calibration ref | LLM vs oracle-best | LLM vs calibration ref |
|---|---|---|---|---|---|
| T | 1–11–18 (0.006) | 1–0–29 | 3–9–18 (0.15) | 9–12–9 (0.66) | 11–10–9 (1.00) |
| W | **19–7–4 (0.029)** | 19–7–4 (0.029) | 19–7–4 (0.029) | 9–15–6 (0.31) | 9–15–6 (0.31) |
| R | **18–5–7 (0.011)** | 18–4–8 (0.004) | 18–4–8 (0.004) | 10–4–16 (0.18) | 10–3–17 (0.092) |
| Pooled | 38–23–29 (0.072) | 38–11–41 (1.4e-4) | — | 28–31–31 (0.79) | — |

**Mean energy saving when jointly feasible:**

| | GP vs oracle-best | LLM vs oracle-best |
|---|---|---|
| T | 1.2% (n = 19) | 9.1% (n = 18) |
| W | 9.8% (n = 27) | 3.6% (n = 21) |
| R | 4.5% (n = 28) | 8.9% (n = 28) |

**Which reference was oracle-best:**
- T: `latency_greedy` in 26 roots, `first_fit` in 2, `graph_headroom` in 2.
- W: `kube_least_allocated` in all 30.
- R: `graph_headroom` in 28, `kube_least_allocated` in 2.

**Kind of decision in GP vs oracle-best:**
- T: 11 losses on feasibility, 18 ties, 1 win on energy.
- W: 19 wins on energy, 4 losses on energy, 3 losses on feasibility, 4 ties.
- R: 18 wins on energy, 3 losses on energy, 2 losses on feasibility, 7 ties.

**Observations.**
- On W and R, GP's wins over the hand-written references are energy wins with both feasible:
  GP found lower-energy programs that held up on fresh seeds.
- On W the oracle reference is the initialization best itself. "GP beat the best hand-written
  reference" and "GP improved on its starting archive" are therefore the same 19–7–4 comparison
  there.
- On T, GP essentially reproduces its initialization. Its 11 losses to the oracle are cases
  where the initialization best was itself infeasible on fresh seeds: initialization best vs
  oracle on T is 0–11–19.
- The LLM's mean energy saving when feasible is larger on T and R. Its lower feasibility on T and
  W cancels this under feasibility-first ordering.

## What this does and does not support

- **Supports, as a post hoc observation:**
  - The LLM-vs-GP direction differed between the historical task and the two new variants (T
    9–3 vs W+R 18–36 decisive; Fisher p = 0.011, unadjusted, chosen after seeing the data).
  - On T, the LLM led GP in direction in every study (A, C, D, E).
  - On T, GP rarely improved on its starting archive in any study.
  - Zero-shot proposals cost less than half as much per root as iterative evolution, with no
    worse outcomes in direction.
- **Does not support:**
  - **Any causal claim.** It cannot show that the LLM's advantage was *caused* by development on
    T, or that it is specific to T. Both operators, the archive and the prompt were developed on
    T. T's many ties (16 same-program roots) make its decisive subset small and selected. The
    contrast falls to p = 0.077 without the 12 rerun roots.
  - **A conclusion that GP is better** on W or R. Those per-family contrasts were not significant
    (W p = 0.061, R p = 0.23) and were descriptive by protocol.
- **Families are related variants, not independent tasks.** T and R share the cluster and
  workload, and W shares the generator. Three families do not give three independent
  replications. The cross-study T sum (30–6) mixes protocols and is not a pooled test.
- **Non-significance does not establish equivalence.** The null registered primary (27–39–24),
  K2 (15–27–48) and the per-family nulls do not show that the operators, or iteration and
  zero-shot, are equally good.
- **Multiplicity.** This report examines many contrasts after the fact. Any one p-value near 0.01
  to 0.05 is expected by chance in a set this size.
