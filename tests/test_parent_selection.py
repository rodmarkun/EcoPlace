"""Parent selection: uniform stays byte-identical, tournament applies real pressure.

At 58.8 work units the 2026-08 campaigns evaluated ~42 candidates across 4 islands of
population 8 with uniform-random parent selection. Pools barely filled once, so almost
everything proposed survived and the loop was parallel random restarts rather than an
evolutionary search. Tournament selection restores pressure; it lives in the *shared*
runner so the LLM and GP arms receive it identically.
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from evolution.campaign import CampaignRunner


def _runner(k: int) -> CampaignRunner:
    """A CampaignRunner shell exposing only what `_choose_parent` reads."""
    runner = CampaignRunner.__new__(CampaignRunner)
    runner.plan = SimpleNamespace(profile=SimpleNamespace(parent_selection_tournament_k=k))
    return runner


def _by_id(entries: dict[str, tuple[bool, float | None, int]]) -> dict[str, dict]:
    """Build the archive shape `_choose_parent` reads: feasible, energy, proposal_seq."""
    out = {}
    for candidate_id, (feasible, energy, seq) in entries.items():
        out[candidate_id] = {
            "proposal_seq": seq,
            "evaluation": {
                "evaluation": {
                    "fitness": {
                        "feasible": feasible,
                        "infeasibility_vector": [0.0, 0.0, 0.0, 0.0],
                        "summed_primitive_shortfall": 0.0,
                        "normalized_energy": energy,
                        "proposal_seq": seq,
                        "n_instances": 1,
                        "n_bundles": 3,
                    }
                }
            },
        }
    return out


def test_k_of_one_reproduces_the_historical_uniform_draw():
    """The default must be bit-identical to `rng.choice`, or every archive drifts."""
    candidates = [f"c{i}" for i in range(8)]
    by_id = _by_id({name: (True, 1.0 + index, index) for index, name in enumerate(candidates)})
    runner = _runner(1)
    for seed in range(50):
        expected = random.Random(seed).choice(candidates)
        assert runner._choose_parent(candidates, by_id, random.Random(seed)) == expected


def test_tournament_never_returns_a_candidate_worse_than_all_it_drew():
    """A k-way tournament must return the best of its own draws, by the Phase-3 order."""
    candidates = [f"c{i}" for i in range(8)]
    energies = {name: 1.0 + index for index, name in enumerate(candidates)}
    by_id = _by_id({name: (True, energies[name], index) for index, name in enumerate(candidates)})
    runner = _runner(3)
    for seed in range(200):
        # Re-derive the same draws the method makes from an identically-seeded rng.
        rng = random.Random(seed)
        drawn = [rng.choice(candidates) for _ in range(3)]
        picked = runner._choose_parent(candidates, by_id, random.Random(seed))
        assert energies[picked] == min(energies[name] for name in drawn)


def test_tournament_biases_toward_better_candidates():
    """Over many draws the best candidate must be selected far more often than uniform."""
    candidates = [f"c{i}" for i in range(8)]
    by_id = _by_id({name: (True, 1.0 + index, index) for index, name in enumerate(candidates)})
    uniform, tournament = _runner(1), _runner(3)
    uniform_best = sum(
        1
        for seed in range(400)
        if uniform._choose_parent(candidates, by_id, random.Random(seed)) == "c0"
    )
    tournament_best = sum(
        1
        for seed in range(400)
        if tournament._choose_parent(candidates, by_id, random.Random(seed)) == "c0"
    )
    assert tournament_best > 2 * uniform_best


def test_infeasible_candidates_rank_below_feasible_ones():
    """The tournament uses the same total order survivor selection uses."""
    candidates = ["feasible", "infeasible"]
    by_id = _by_id({"feasible": (True, 5.0, 1), "infeasible": (False, None, 0)})
    runner = _runner(2)
    # Whatever the draws, an infeasible candidate can only win if it was drawn twice.
    for seed in range(100):
        rng = random.Random(seed)
        drawn = [rng.choice(candidates) for _ in range(2)]
        picked = runner._choose_parent(candidates, by_id, random.Random(seed))
        assert picked == ("infeasible" if set(drawn) == {"infeasible"} else "feasible")


def test_unevaluated_candidates_rank_last_deterministically():
    candidates = ["done", "pending"]
    by_id = _by_id({"done": (True, 9.0, 1)})
    by_id["pending"] = {"proposal_seq": 2, "evaluation": None}
    runner = _runner(2)
    for seed in range(100):
        rng = random.Random(seed)
        drawn = [rng.choice(candidates) for _ in range(2)]
        picked = runner._choose_parent(candidates, by_id, random.Random(seed))
        assert picked == ("pending" if set(drawn) == {"pending"} else "done")


def test_single_candidate_pool_is_returned_without_drawing():
    runner = _runner(3)
    pool = _by_id({"only": (True, 1.0, 0)})
    assert runner._choose_parent(["only"], pool, random.Random(0)) == "only"
