**Study E: multi-family confirmatory study — internally frozen protocol, 5 October 2026**

**Motivation, disclosed.** Studies A–D (all sealed and analysed before this protocol) found:
an LLM-over-GP direction in every study on the historical task, significant only in the
registered four-seed study C; that single-seed selection yields programs at the SLO boundary;
and that, on C's roots, four-seed search was associated with much higher fresh-seed feasibility
for both operators. Reviewers can rightly object that the confirmatory evidence rests on one
study of one task. Study E is designed after those results to test the LLM advantage at scale,
across three task families, under the multi-seed protocol, against stronger baselines. Its
design choices (families, protocol, baselines) were informed by A–D; its outcomes were not
known when it was frozen.

**Hypotheses.** Under the multi-seed protocol and matched controls:
(H1) LLM-driven search finds better placement policies than same-representation GP;
(H2) the LLM champion is better than the best hand-written reference policy for the root;
(H3) iterative LLM-driven evolution is better than one-step LLM proposals from the starting
archive without iteration (zero-shot).

**Task families.** Three related task variants from the same sealed generator
(`sim/scenarios/dev_tight.py`), 30 new paired roots each (90 roots). They are not independent
task families: T and R share the cluster and workload, and W shares the generator and its
calibration machinery. Breadth claims are limited accordingly.
- **T** `dev-continuum-tight`: the historical 15-node task (studies A, C, D);
- **W** `dev-continuum-wide`: a 17-node cluster with a different topology and site structure;
- **R** `dev-continuum-drift`: the historical cluster with three nodes cordoned from 150 s to
  330 s (straddling the flash crowd) and a sustained 10% demand step.

**Headroom gate (checked before freezing; `calibration/`).** The nine references were scored
on four dedicated calibration seeds per family (never reused) across all three bundles. The gate,
as checked: (a) at least one reference feasible on every calibration instance; (b) at least one
other reference infeasible only narrowly (summed primitive shortfall below 0.5), so a feasible,
differently placed policy plausibly exists. Infeasible references receive no energy value by
design, so the gate does not measure the energy of near-feasible alternatives. T:
`latency_greedy`/`network_overhead` feasible (energy 0.83), `first_fit` shortfall 0.002; W: only
`kube_least_allocated` feasible (worst SLO 0.900, energy 1.32), `least_allocated` shortfall 0.16;
R: only `graph_headroom` feasible (energy 1.28), `latency_greedy` shortfall 0.013. All three
passed.

**Treatments.** Inherited byte-for-byte from Study C (`INHERITANCE.json`): ladder, islands,
feedback profile, GP configuration, LLM engine (DeepSeek V4.1 Flash, low reasoning, 32,000
output tokens, temperature 0.7, top-p 0.8, DeepSeek-only routing), 56-unit work ceiling,
candidate safety ceiling, selection rule, paired initialization barrier, and the multi-seed
protocol (`evaluation_replicates: 4`; search seeds `derive_seed(root,'sim','scenario',id,
'replicate',j)`, j = 0..3). Task files differ from Study C's only in the scenario. Archives use
Study A/C's procedure unchanged (seven references, five seeded mutants). Each family's
calibration-feasible reference is in the archive, but initialization records are scored under
alternating island constraints (e.g. W's reference falls on the stricter qos-tight island), so a
feasible starting record is not guaranteed.

**Zero-shot LLM arm (no iteration).** Per root, up to 20 proposals from the same operator,
engine, proposal modes, retry and grammar-aware-repair rules as the evolving LLM arm. Each
proposal's parent is drawn uniformly from the 12 initialization programs, and its inspirations
uniformly from the remaining 11 (up to the profile's maximum), using `zeroshot_seed`. Parents
are built from the LLM arm's initialization records exactly as the campaign builds them, including
their text feedback; the zero-shot arm never sees any of its own proposals or their scores
(grammar-aware repair necessarily shows the model its own validation error). Valid proposals are
evaluated through the same ladder on the root's search grid, rotating island constraints
(proposal k on island k mod 2) as the campaign does, followed by the same archive assessment.
The campaign's behavioural-novelty screen is not applied; it can only let more zero-shot
candidates reach L2. The zero-shot champion is the best feasibility-first sort key over the
initialization records and the zero-shot L2 records, with the same rule as the arms. Zero-shot
spend is capped at $0.20 per root, and the stage may stop early at the cap; N denotes the number
of proposal slots it used.

**Final evaluation.** Labels: LLM champion, GP champion, initialization best, zero-shot
champion, the LLM arm's champion restricted to its first N adaptive proposal slots
(`llm_champion_atN`, the equal-opportunity comparator for zero-shot; both arms allow at most three
attempts per slot), and the nine references. Each is
scored on eight fresh final seeds per scenario across all three bundles under the base
constraints. Seeds are disjoint from every seed used before (`SEEDS.json`). Selections are
sealed before any final grid is run. For family T roots, every label is additionally scored on
eight fresh `dev-continuum-drift` seeds (descriptive transfer).

**Primary endpoint and test.** P: LLM champion versus GP champion, pooled over the 90 roots,
feasibility-first directions of the frozen statistics module (sha256 `b42bd4fd…`, tolerance
1e-9). Exact two-sided sign test at α = 0.05. A directional claim requires p ≤ 0.05 and more LLM
wins than losses.

**Key secondary endpoints (gatekept).** Tested only if P supports the LLM. Then K1 and K2 are
tested with Holm's procedure over two tests at α = 0.05 (frozen `holm_two`):
- K1: LLM champion versus the root's **oracle-best reference**, the reference with the best
  feasibility-first outcome on that root's final grid among the nine (chosen with hindsight,
  which makes K1 conservative against the LLM);
- K2: `llm_champion_atN` versus zero-shot champion: iterative evolution against one-step
  proposals from the same archive with the same number of proposal slots. K2 is the
  iteration-specific test; the full LLM champion (56 work units, $0.75 cap) against zero-shot
  (at most 20 slots, $0.20 cap) compares differently budgeted packages and is descriptive only.

If P is not supported, K1 and K2 are reported descriptively only.

**Descriptive (unadjusted).** Per-family P, K1, K2; GP versus oracle-best reference; full LLM
champion versus zero-shot; each champion versus initialization best; final-grid feasibility and
mean energy when feasible; jointly feasible energy savings with bootstrap intervals
(`analysis` seeds in `SEEDS.json`); transfer of T labels to drift; spend, requests and work.

**Execution.** Study C/D launcher with: registration directory, three families interleaved in
index order (T00, W00, R00, T01, …), six pair slots, both evolving arms and the zero-shot arm
running concurrently after the pair certificate, per-root API caps of $0.75 (evolving LLM arm;
the highest per-root spend in C was $0.56) and $0.20 (zero-shot), and a project ceiling of $145
set by the investigator. Prior spend plus all caps is at most $143.87, so the global ceiling
cannot bind. Settled no-response and HTTP errors, connectivity and peak-window waits, memory
back-pressure, a disk-space guard, crash recovery and reconciliation are inherited; replayed
requests whose global reservation was never settled are settled from the journal. A Codex
pre-launch review (`docs/reviews/CODEX_STUDY_E_DESIGN_REVIEW_20261005.md`) found four blockers
and several should-fix items; all were addressed before this freeze. All 90 roots
stay in the ledger; missing primary observations block confirmatory inference; no journal
editing and no replacement roots.
