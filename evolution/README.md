# evolution/ — search harness

Evolutionary search over PlacementScript policies, built on a pinned fork of
[ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve). The fork is installed editable from
`../ShinkaEvolve`; `shinkaevolve.lock.json` and the hash-pinned patches in `shinkaevolve-patches/`
rebuild its exact tree (see the root README).

| Module | Role |
|---|---|
| `task/` | ShinkaEvolve task package: initial policies and `evaluate.py` |
| `evaluation.py`, `ladder.py` | Policy evaluation and the deterministic L0 → L1 → L2 evaluation ladder (cheap analytical screen, short episodes, full episodes) with epsilon audits of rejected candidates |
| `islands.py`, `profiles.py` | Constraint-tightness islands and fail-closed evolution profiles |
| `gp_operator.py` | Grammar-guided GP proposal operator over the PlacementScript AST |
| `llm_operator.py`, `operators.py` | LLM proposal operator and the proposal controls shared by both arms |
| `openrouter_engine.py`, `codex_engine.py` | LLM transports: OpenRouter (DeepSeek) and the Codex CLI (gpt-6.1-sol), with request journals and spend accounting |
| `model.py` | Model pins and the API cost ledger |
| `feedback.py` | Structured simulator feedback (separate ablation profile only) |
| `campaign.py`, `run_campaign.py` | Resumable, append-only campaign runner and its CLI |
| `g0.py`, `g_ablate.py` | Throughput pilots and the paired determinism check |
| `risk_gates/` | Property-based checks (Hypothesis) on policies: totality, determinism, batch/scalar equivalence, decision-latency budget, proposer budget compliance |

Selection uses the lexicographic fitness of `sim.energy.fitness` (feasible before infeasible, then
modeled energy); `combined_score` exists only for ShinkaEvolve's dashboards.
