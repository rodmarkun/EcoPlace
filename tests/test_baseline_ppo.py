from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.ppo import (
    DECISION_STEP_UNIT,
    FrozenPPOProposer,
    GraphEncodingConfig,
    GraphObservationEncoder,
    PlacementEpisode,
    PPOCorpus,
    PPOHyperparameters,
    PPOTuningBudget,
    _make_training_env,
    corpus_from_snapshot_library,
    frozen_ppo_bundle,
    load_frozen_ppo_bundle,
    load_tuning_budget,
    tune_and_train_frozen_ppo,
)

ROOT = Path(__file__).parents[1]


class FakeModel:
    artifact_sha256 = "a" * 64

    def __init__(self) -> None:
        self.calls = 0

    def action_logits(self, observation: np.ndarray) -> np.ndarray:
        del observation
        self.calls += 1
        return np.asarray([0.25, 1.5, -2.0, -4.0], dtype=np.float64)


def _ctx() -> dict:
    nodes = [
        {
            "id": 0,
            "tier": 0,
            "cpu_capacity": 4000,
            "cpu_allocated": 1000,
            "mem_capacity": 8000,
            "mem_allocated": 1000,
            "power_idle_w": 80.0,
            "power_max_w": 200.0,
            "suspended": False,
            "suspendable": False,
            "latency_to_ms": [0.0, 20.0, 40.0],
        },
        {
            "id": 1,
            "tier": 2,
            "cpu_capacity": 2000,
            "cpu_allocated": 0,
            "mem_capacity": 4000,
            "mem_allocated": 0,
            "power_idle_w": 4.0,
            "power_max_w": 10.0,
            "suspended": False,
            "suspendable": True,
            "latency_to_ms": [20.0, 0.0, 5.0],
        },
        {
            "id": 2,
            "tier": 2,
            "cpu_capacity": 2000,
            "cpu_allocated": 0,
            "mem_capacity": 4000,
            "mem_allocated": 0,
            "power_idle_w": 4.0,
            "power_max_w": 10.0,
            "suspended": True,
            "suspendable": True,
            "latency_to_ms": [40.0, 5.0, 0.0],
        },
    ]
    pods = [
        {
            "id": 0,
            "service": 0,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": -1,
        },
        {
            "id": 1,
            "service": 1,
            "app": 0,
            "cpu_request": 250,
            "mem_request": 128,
            "stateful": False,
            "has_pdb": False,
            "image_mb": 10,
            "current_node": 1,
        },
    ]
    graph = {"src": [0], "dst": [1], "rate": [5.0], "payload_kb": [2.0]}
    return {"nodes": nodes, "pods": pods, "graph": graph, "now_ms": 0}


def _episode(source: str, split: str, *, data_role: str = "quarantined-pilot") -> PlacementEpisode:
    ctx = _ctx()
    if source != "a":
        ctx["nodes"][0]["cpu_allocated"] += 1
    return PlacementEpisode(
        episode_id=f"episode-{source}",
        source_library_sha256=source * 64,
        source_episode_sha256=(source.upper().lower()) * 64,
        data_role=data_role,
        split_role=split,
        decision_start_step=0,
        ctx=ctx,
        pending_pod_ids=(0,),
    )


def _corpus(source: str, split: str, *, data_role: str = "quarantined-pilot") -> PPOCorpus:
    return PPOCorpus(data_role, split, (_episode(source, split, data_role=data_role),))


def test_graph_encoding_has_candidate_mask_and_neighbour_signal() -> None:
    ctx = _ctx()
    config = GraphEncodingConfig(max_nodes=3)
    encoder = GraphObservationEncoder(config)
    observation = encoder.encode(ctx["pods"][0], ctx["nodes"][:2], ctx)
    rows = observation.reshape(3, -1)
    assert encoder.candidate_mask(observation).tolist() == [True, True, False]
    assert rows[0, 9] > rows[1, 9]
    assert rows[0, 10] > rows[1, 10]


def test_frozen_ppo_is_artifact_bound_and_uses_one_counterfactual_model_call() -> None:
    model = FakeModel()
    bundle = frozen_ppo_bundle(
        model,
        GraphEncodingConfig(max_nodes=3),
        training_receipt_sha256="b" * 64,
        registration_status="development-provisional",
    )
    ctx = _ctx()
    assert bundle.scorer.score_batch(ctx["pods"][0], ctx["nodes"][:2], ctx) == [0.25, 1.5]
    assert bundle.artifact_sha256 == model.artifact_sha256
    state = {"nodes": ctx["nodes"], "pods": ctx["pods"]}
    state["pods"][0]["current_node"] = 0
    state["nodes"][0]["cpu_allocated"] += 250
    state["nodes"][0]["mem_allocated"] += 128
    before = model.calls
    moves = FrozenPPOProposer(bundle.scorer).propose_moves(state, ctx["graph"], 1)
    assert moves == [{"kind": 0, "pod": 0, "dest": 1}]
    assert model.calls == before + 1


def test_tuning_budget_is_explicit_bounded_and_provisional() -> None:
    budget, encoding = load_tuning_budget(ROOT / "baselines" / "ppo_tuning.json")
    assert not budget.confirmatory_ready
    assert encoding.max_nodes == 64
    assert sum(trial.total_timesteps for trial in budget.trials) == budget.max_total_timesteps
    with pytest.raises(ValueError, match="exceeds"):
        PPOTuningBudget("development-provisional", "test", "quarantined-pilot", budget.trials, 1)


def test_snapshot_training_data_must_have_sealed_role() -> None:
    class Library:
        meta = {"data_role": "confirmatory-private"}

    with pytest.raises(ValueError, match="data_role"):
        corpus_from_snapshot_library(Library(), split_role="train")  # type: ignore[arg-type]


def test_online_snapshot_history_requires_sealed_nonoverlapping_decision_ranges() -> None:
    class Snapshot:
        pending_dense_ids = (0,)

        def to_dict(self):
            return {"ctx": _ctx()}

    class Library:
        snapshots = (Snapshot(), Snapshot())
        meta = {
            "data_role": "online-history-before-step",
            "episode_sha256": "c" * 64,
            "decision_step_unit": DECISION_STEP_UNIT,
        }

        def digest(self):
            return "d" * 64

    with pytest.raises(ValueError, match="sealed decision_step_range"):
        corpus_from_snapshot_library(Library(), split_role="train")  # type: ignore[arg-type]

    library = Library()
    library.meta = {
        **library.meta,
        "decision_step_ranges": [
            {"start": 4, "end_exclusive": 5},
            {"start": 4, "end_exclusive": 5},
        ],
    }
    with pytest.raises(ValueError, match="ordered, and non-overlapping"):
        corpus_from_snapshot_library(library, split_role="train")  # type: ignore[arg-type]

    library.meta["decision_step_ranges"][1] = {"start": 7, "end_exclusive": 8}
    corpus = corpus_from_snapshot_library(library, split_role="train")  # type: ignore[arg-type]
    assert [episode.decision_start_step for episode in corpus.episodes] == [4, 7]


def test_training_rejects_overlapping_source_provenance_before_optional_import(
    tmp_path: Path,
) -> None:
    budget = PPOTuningBudget(
        "development-provisional",
        "test",
        "quarantined-pilot",
        (PPOHyperparameters(0.001, 8, 8, 1, 0.9, 0.0, (8,), 8),),
        8,
    )
    with pytest.raises(ValueError, match="disjoint"):
        tune_and_train_frozen_ppo(
            _corpus("a", "train"),
            _corpus("a", "validation"),
            budget,
            GraphEncodingConfig(max_nodes=3),
            tmp_path,
            root_seed=1,
        )


def test_training_rejects_identical_content_with_different_provenance_labels(
    tmp_path: Path,
) -> None:
    budget = PPOTuningBudget(
        "development-provisional",
        "test",
        "quarantined-pilot",
        (PPOHyperparameters(0.001, 8, 8, 1, 0.9, 0.0, (8,), 8),),
        8,
    )
    training = _corpus("a", "train")
    relabeled = dataclasses.replace(
        _episode("b", "validation"),
        ctx=training.episodes[0].ctx,
    )
    validation = PPOCorpus("quarantined-pilot", "validation", (relabeled,))
    with pytest.raises(ValueError, match="trajectory content"):
        tune_and_train_frozen_ppo(
            training,
            validation,
            budget,
            GraphEncodingConfig(max_nodes=3),
            tmp_path,
            root_seed=1,
        )


def test_training_environment_is_action_conditioned_and_masks_invalid_nodes() -> None:
    ctx = _ctx()
    ctx["pods"][1]["current_node"] = -1
    episode = dataclasses.replace(_episode("a", "train"), ctx=ctx, pending_pod_ids=(0, 1))
    corpus = PPOCorpus("quarantined-pilot", "train", (episode,))
    encoder = GraphObservationEncoder(GraphEncodingConfig(max_nodes=3))
    env = _make_training_env(corpus, encoder, seed=3)
    with pytest.raises(TypeError, match="seed"):
        env.reset()
    with pytest.raises(ValueError, match="explicit int"):
        env.reset(seed=None)
    before, _ = env.reset(seed=3)
    mask = env.action_masks()
    assert mask.tolist() == [True, True, False, False]
    after, _reward, terminated, _truncated, _info = env.step(1)
    assert not terminated
    assert not np.array_equal(before, after)
    # The second decision observes pod 0 on node 1 and the added allocation.
    rows = after.reshape(3, -1)
    assert rows[1, 1] > before.reshape(3, -1)[1, 1]
    assert rows[0, 9] > 0
    env.close()


def test_optional_maskable_ppo_tiny_training_writes_fail_closed_receipt(tmp_path: Path) -> None:
    pytest.importorskip("sb3_contrib")
    budget = PPOTuningBudget(
        "development-provisional",
        "test",
        "quarantined-pilot",
        (PPOHyperparameters(0.001, 8, 8, 1, 0.9, 0.0, (8,), 16),),
        16,
    )
    receipt = tune_and_train_frozen_ppo(
        _corpus("a", "train"),
        _corpus("b", "validation"),
        budget,
        GraphEncodingConfig(max_nodes=3),
        tmp_path,
        root_seed=17,
    )
    assert json.loads((tmp_path / "training_receipt.json").read_text()) == receipt
    assert receipt["training_corpus_sha256"] != receipt["validation_corpus_sha256"]
    with pytest.raises(ValueError, match="P2-frozen"):
        load_frozen_ppo_bundle(tmp_path, expected_receipt_sha256=receipt["sha256"])
    loaded = load_frozen_ppo_bundle(
        tmp_path,
        expected_receipt_sha256=receipt["sha256"],
        require_p2_frozen=False,
    )
    ctx = _ctx()
    scores = loaded.scorer.score_batch(ctx["pods"][0], ctx["nodes"][:2], ctx)
    assert len(scores) == 2
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="externally pinned"):
        load_frozen_ppo_bundle(
            tmp_path,
            expected_receipt_sha256="f" * 64,
            require_p2_frozen=False,
        )
    malicious = json.loads(json.dumps(receipt))
    malicious["champion_artifact"] = "../escape.zip"
    malicious["trials"][0]["artifact"] = "../escape.zip"
    body = {key: value for key, value in malicious.items() if key != "sha256"}
    malicious["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (tmp_path / "training_receipt.json").write_text(
        json.dumps(malicious, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ValueError, match="safe basename"):
        load_frozen_ppo_bundle(
            tmp_path,
            expected_receipt_sha256=malicious["sha256"],
            require_p2_frozen=False,
        )
