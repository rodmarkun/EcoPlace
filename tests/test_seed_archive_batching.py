"""Population parity and batched proposals must work together.

`--seed-archive` (population parity) wraps `CampaignRunner` in a subclass that serves a
fixed initialization archive before delegating to the real proposer. `--proposals-per-call`
(batching) hands the runner a pre-fetched sibling through `_proposal(..., initial_offspring=)`.

The two features were built at different times and never exercised together. The override
did not accept the keyword, so the first campaign that combined them died with
`TypeError: _proposal() got an unexpected keyword argument 'initial_offspring'` the moment
the first batched candidate arrived — after the archive was exhausted, which is why every
deterministic arm finished normally (they do not batch) and only the LLM arm was lost.

These tests pin the contract rather than the symptom: the override must accept the
keyword, forward it, and still serve the archive first.
"""

from __future__ import annotations

import inspect
from typing import Any

from evolution.campaign import CampaignRunner
from experiments.allbaselines_harness.initialization import seeded_campaign_runner

ARCHIVE = {"payload_sha256": "f" * 64, "sources": ["policy-a", "policy-b"]}


def test_override_signature_matches_the_base_runner():
    """A subclass that narrows its parent's signature is a latent crash."""
    base = inspect.signature(CampaignRunner._proposal)
    override = inspect.signature(seeded_campaign_runner(ARCHIVE)._proposal)
    assert set(base.parameters) == set(override.parameters)
    for name, parameter in base.parameters.items():
        assert override.parameters[name].kind == parameter.kind


def test_archive_entries_are_served_before_the_proposer_runs():
    cls = seeded_campaign_runner(ARCHIVE)
    runner = cls.__new__(cls)
    for seq, expected in enumerate(ARCHIVE["sources"]):
        result = cls._proposal(runner, {"proposal_seq": seq})
        assert result["valid"] and result["source"] == expected
        assert result["metadata"]["operator"] == "fixed_seed_archive"
        assert result["metadata"]["seed_archive_index"] == seq


def test_the_keyword_is_forwarded_once_the_archive_is_exhausted(monkeypatch):
    """Past the archive the sibling must reach the base runner, not be swallowed."""
    seen: dict[str, Any] = {}

    def fake(self, created, *, initial_offspring=None):
        seen["seq"] = created["proposal_seq"]
        seen["initial_offspring"] = initial_offspring
        return {"valid": True, "source": "delegated", "error": None, "metadata": {}}

    monkeypatch.setattr(CampaignRunner, "_proposal", fake)
    cls = seeded_campaign_runner(ARCHIVE)
    runner = cls.__new__(cls)
    sentinel = object()
    beyond = len(ARCHIVE["sources"])
    result = cls._proposal(runner, {"proposal_seq": beyond}, initial_offspring=sentinel)
    assert result["source"] == "delegated"
    assert seen["seq"] == beyond
    assert seen["initial_offspring"] is sentinel


def test_the_keyword_is_accepted_while_the_archive_is_still_serving(monkeypatch):
    """Passing a sibling for an archive index must not raise, even though it is unused."""
    monkeypatch.setattr(CampaignRunner, "_proposal", lambda *a, **k: {})
    cls = seeded_campaign_runner(ARCHIVE)
    runner = cls.__new__(cls)
    result = cls._proposal(runner, {"proposal_seq": 0}, initial_offspring=object())
    assert result["source"] == ARCHIVE["sources"][0]
