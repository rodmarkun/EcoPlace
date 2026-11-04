"""Effective bundle JSON must preserve structured energy provenance and identity."""

import dataclasses
import json

import pytest

from evolution.evaluation import config_for_development_bundle
from sim.energy import build_development_bundles, load_pinned_sources
from sim.spec import (
    EnergyCoefficients,
    EnergySourceRecord,
    config_sha256,
    episode_config_from_json,
    episode_config_to_json,
    load_episode_config,
)


@pytest.fixture(scope="module")
def base():
    return load_episode_config("sim/scenarios/dev/continuum_tight.json")


@pytest.mark.parametrize("bundle_index", range(3))
def test_every_effective_bundle_retains_exact_identity(base, bundle_index):
    bundle = build_development_bundles(load_pinned_sources())[bundle_index]
    config = config_for_development_bundle(base, bundle)
    encoded = episode_config_to_json(config)
    restored = episode_config_from_json(encoded)
    assert episode_config_to_json(restored) == encoded
    assert config_sha256(restored) == config_sha256(config)
    assert restored.energy.sha256() == config.energy.sha256()
    assert restored.energy.source_records == config.energy.source_records
    assert len(restored.energy.source_records) == 5
    assert all(isinstance(r, EnergySourceRecord) for r in restored.energy.source_records)


@pytest.mark.parametrize("omit", [None, "source_records", "energy"])
def test_empty_records_and_legacy_defaults(base, omit):
    config = dataclasses.replace(base, energy=EnergyCoefficients())
    raw = json.loads(episode_config_to_json(config))
    if omit == "energy":
        raw.pop("energy")
    if omit == "source_records":
        raw["energy"].pop("source_records")
    restored = episode_config_from_json(json.dumps(raw))
    assert restored == config
    assert restored.energy.source_records == ()


@pytest.mark.parametrize(
    "damage", ["missing", "unknown", "empty", "unattributed", "not_object", "null"]
)
def test_malformed_source_records_are_rejected(base, damage):
    bundle = build_development_bundles(load_pinned_sources())[0]
    raw = json.loads(episode_config_to_json(config_for_development_bundle(base, bundle)))
    record = raw["energy"]["source_records"][0]
    if damage == "missing":
        record.pop("entry_id")
    if damage == "unknown":
        record["unexpected_field"] = "invalid"
    if damage == "empty":
        record["entry_id"] = ""
    if damage == "unattributed":
        record.update(url=None, doi=None)
    if damage == "not_object":
        raw["energy"]["source_records"][0] = False
    if damage == "null":
        raw["energy"]["source_records"] = None
    with pytest.raises((TypeError, ValueError)):
        episode_config_from_json(json.dumps(raw))
