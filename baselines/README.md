# baselines/ — comparison methods

Every comparator is a `PolicyBundleAdapter` (`adapter.py`): a batch scorer, a rebalancing
proposer, an immutable identity and canonical metadata. Baselines are evaluated by the same
simulator, evaluation ladder, scenario grid and constraints as the evolved policies.

| Module | Method |
|---|---|
| `handwritten.py` | Hand-written heuristics: latency-greedy, energy-greedy, first fit, least/most allocated, network overhead (PlacementScript sources in `runtime/policies/`) |
| `same_dsl_gp.py` | Entry point for the grammar-guided GP arm over PlacementScript (`evolution/gp_operator.py`) |
| `gphh.py` | Classic strongly typed GP hyper-heuristic |
| `online_gphh.py` | Online GPHH that re-evolves at drift triggers under a per-trigger budget |
| `random_ast.py` | Unguided random AST mutation (search floor) |
| `ppo.py`, `ppo_retraining.py` | Graph-aware PPO policy with sealed training provenance, and periodic retraining (needs `--extra rl`) |
| `assignment_oracles.py` | Per-instance assignment oracles used as reference points |

Tuning budgets and frozen configurations are in the `*.json` files next to the modules.
