"""Loader for the sealed Phase-3 energy-analysis registration.

The canonical bundle builders are used when this artifact is prepared, not when a
confirmatory result is authorized.  Scientific consumers load the repository
artifact below and recompute its canonical-JSON hashes.  This keeps bundle builders
as construction helpers while the registered bundle identities, role rosters, and
complete GSA strata payload remain the authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_REGISTRATION_PATH = Path(__file__).with_name("data") / "energy_analysis_registration.json"
_REGISTRATION_SCHEMA = "energy-analysis-registration-v1"
_SPLIT_SCHEMA = "energy-bundle-split-registration-v1"
_GSA_SCHEMA = "gsa-strata-design-registration-v1"
_HEX = frozenset("0123456789abcdef")


def canonical_json_sha256(payload: object) -> str:
    """Hash strict canonical JSON (sorted keys, compact separators, no NaN)."""
    encoded = json.dumps(
        _thaw_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return types.MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sequence(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _bundle_identities(
    name: str, value: object
) -> tuple[tuple[str, str], ...]:
    records = _sequence(name, value)
    identities: list[tuple[str, str]] = []
    for index, raw in enumerate(records):
        record = _mapping(f"{name}[{index}]", raw)
        if set(record) != {"bundle_id", "manifest_sha256"}:
            raise ValueError(
                f"{name}[{index}] must contain only bundle_id and manifest_sha256"
            )
        bundle_id = record["bundle_id"]
        digest = record["manifest_sha256"]
        if not isinstance(bundle_id, str) or not bundle_id:
            raise ValueError(f"{name}[{index}].bundle_id must be a non-empty string")
        if not _is_digest(digest):
            raise ValueError(f"{name}[{index}].manifest_sha256 must be a SHA-256 digest")
        identities.append((bundle_id, digest))
    ids = [bundle_id for bundle_id, _ in identities]
    if not identities or len(ids) != len(set(ids)):
        raise ValueError(f"{name} must contain unique non-empty bundle ids")
    return tuple(identities)


@dataclass(frozen=True)
class EnergyAnalysisRegistration:
    """Immutable, content-sealed split and GSA-design registration."""

    split_payload: Mapping[str, object]
    split_payload_sha256: str
    gsa_design_payload: Mapping[str, object]
    design_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "split_payload", _freeze_json(dict(self.split_payload)))
        object.__setattr__(
            self,
            "gsa_design_payload",
            _freeze_json(dict(self.gsa_design_payload)),
        )
        self.verify_seal()
        self._validate_structure()

    def verify_seal(self) -> None:
        """Recompute both registered hashes over their complete canonical payloads."""
        if not _is_digest(self.split_payload_sha256):
            raise ValueError("split payload_sha256 must be a SHA-256 digest")
        if not _is_digest(self.design_sha256):
            raise ValueError("GSA design_sha256 must be a SHA-256 digest")
        split_payload = _thaw_json(self.split_payload)
        if canonical_json_sha256(split_payload) != self.split_payload_sha256:
            raise ValueError(
                "energy split registration seal mismatch — role rosters, bundle ids, "
                "or manifest hashes differ from the registered payload"
            )
        design_payload = _thaw_json(self.gsa_design_payload)
        if canonical_json_sha256(design_payload) != self.design_sha256:
            raise ValueError(
                "GSA design registration seal mismatch — strata, nuisances, or "
                "development bundle identities differ from the registered payload"
            )

    def _validate_structure(self) -> None:
        if self.split_payload.get("schema") != _SPLIT_SCHEMA:
            raise ValueError("energy split registration has an unexpected schema")
        if self.gsa_design_payload.get("schema") != _GSA_SCHEMA:
            raise ValueError("GSA design registration has an unexpected schema")

        roles_raw = _sequence("split.roles", self.split_payload.get("roles"))
        roles: dict[str, Mapping[str, object]] = {}
        for index, raw in enumerate(roles_raw):
            role = _mapping(f"split.roles[{index}]", raw)
            if set(role) != {"role", "node_class_roster", "bundles"}:
                raise ValueError(
                    "each split role must contain only role, node_class_roster, and bundles"
                )
            role_name = role["role"]
            if not isinstance(role_name, str) or not role_name:
                raise ValueError("split role names must be non-empty strings")
            if role_name in roles:
                raise ValueError("split role names must be unique")
            roster = _sequence(f"split role {role_name} roster", role["node_class_roster"])
            if (
                not roster
                or any(not isinstance(name, str) or not name for name in roster)
                or len(roster) != len(set(roster))
                or tuple(sorted(roster)) != roster
            ):
                raise ValueError(
                    f"split role {role_name} roster must be sorted, unique, and non-empty"
                )
            _bundle_identities(f"split role {role_name} bundles", role["bundles"])
            roles[role_name] = role
        expected_roles = {"development", "held-out-structural"}
        if set(roles) != expected_roles:
            raise ValueError(
                f"split registration roles must be exactly {sorted(expected_roles)}"
            )
        if (
            tuple(roles["development"]["node_class_roster"])
            != tuple(roles["held-out-structural"]["node_class_roster"])
        ):
            raise ValueError(
                "development and held-out registrations must share one complete "
                "node-class roster"
            )

        design_bundles = _bundle_identities(
            "gsa_design.development_bundles",
            self.gsa_design_payload.get("development_bundles"),
        )
        if design_bundles != self.bundle_manifests("development"):
            raise ValueError(
                "GSA design bundle identities must exactly match the registered "
                "development split"
            )
        strata = _sequence("gsa_design.strata", self.gsa_design_payload.get("strata"))
        strata_ids: list[str] = []
        strata_bundles: list[tuple[str, str]] = []
        registered_name_order: tuple[str, ...] | None = None
        for index, raw in enumerate(strata):
            stratum = _mapping(f"gsa_design.strata[{index}]", raw)
            if set(stratum) != {
                "stratum_id",
                "nuisances",
                "bundle_manifest_sha256",
            }:
                raise ValueError(
                    "each registered stratum must contain its id, complete nuisance "
                    "list, and bundle manifest hash"
                )
            stratum_id = stratum["stratum_id"]
            manifest = stratum["bundle_manifest_sha256"]
            if not isinstance(stratum_id, str) or not stratum_id:
                raise ValueError("registered stratum ids must be non-empty strings")
            if not _is_digest(manifest):
                raise ValueError("registered stratum bundle hash must be a SHA-256 digest")
            nuisances = _sequence(
                f"gsa_design.strata[{index}].nuisances", stratum["nuisances"]
            )
            if not nuisances:
                raise ValueError("registered strata must enumerate every nuisance")
            nuisance_names: list[str] = []
            for nuisance_raw in nuisances:
                nuisance = _mapping("registered nuisance", nuisance_raw)
                required = {
                    "name",
                    "low",
                    "high",
                    "conditional_basis",
                    "within_stratum_independence",
                    "distribution",
                }
                if set(nuisance) != required:
                    raise ValueError(
                        "registered nuisances must contain the complete conditional definition"
                    )
                name = nuisance["name"]
                if not isinstance(name, str) or not name:
                    raise ValueError("registered nuisance names must be non-empty strings")
                low, high = nuisance["low"], nuisance["high"]
                if (
                    isinstance(low, bool)
                    or isinstance(high, bool)
                    or not isinstance(low, int | float)
                    or not isinstance(high, int | float)
                    or not math.isfinite(low)
                    or not math.isfinite(high)
                    or low >= high
                ):
                    raise ValueError(
                        f"registered nuisance {name!r} must have finite low < high"
                    )
                for field_name in (
                    "conditional_basis",
                    "within_stratum_independence",
                ):
                    statement = nuisance[field_name]
                    if not isinstance(statement, str) or len(statement.strip()) < 8:
                        raise ValueError(
                            f"registered nuisance {name!r} {field_name} must be "
                            "meaningful string prose"
                        )
                distribution = nuisance["distribution"]
                if distribution not in {"uniform", "loguniform"}:
                    raise ValueError(
                        f"registered nuisance {name!r} has unsupported distribution"
                    )
                if distribution == "loguniform" and low <= 0:
                    raise ValueError(
                        f"registered nuisance {name!r} loguniform low must be positive"
                    )
                nuisance_names.append(name)
            if len(nuisance_names) != len(set(nuisance_names)):
                raise ValueError("registered nuisance names must be unique within a stratum")
            name_order = tuple(nuisance_names)
            if registered_name_order is None:
                registered_name_order = name_order
            elif name_order != registered_name_order:
                raise ValueError(
                    "every registered stratum must expose the same ordered nuisance set"
                )
            strata_ids.append(stratum_id)
            strata_bundles.append((stratum_id, manifest))
        if len(strata_ids) != len(set(strata_ids)):
            raise ValueError("registered stratum ids must be unique")
        if tuple(strata_bundles) != design_bundles:
            raise ValueError(
                "registered strata must be ordered as and bound to every development bundle"
            )

    def _role(self, role_name: str) -> Mapping[str, object]:
        roles = _sequence("split.roles", self.split_payload["roles"])
        for raw in roles:
            role = _mapping("split role", raw)
            if role["role"] == role_name:
                return role
        raise ValueError(f"unregistered energy bundle role {role_name!r}")

    def bundle_manifests(self, role_name: str) -> tuple[tuple[str, str], ...]:
        """Registered ``(bundle_id, manifest_sha256)`` pairs in canonical order."""
        role = self._role(role_name)
        return _bundle_identities(f"split role {role_name} bundles", role["bundles"])

    def node_class_roster(self, role_name: str) -> tuple[str, ...]:
        """The complete registered node-class roster for one role."""
        role = self._role(role_name)
        return tuple(_sequence(f"split role {role_name} roster", role["node_class_roster"]))

    def strata_payloads(self) -> tuple[Mapping[str, object], ...]:
        """Immutable complete registered stratum payloads in canonical order."""
        raw = _sequence("gsa_design.strata", self.gsa_design_payload["strata"])
        return tuple(_mapping("registered stratum", item) for item in raw)


def _load_energy_analysis_registration(path: Path) -> EnergyAnalysisRegistration:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load energy analysis registration {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != _REGISTRATION_SCHEMA:
        raise ValueError(f"{path}: unexpected energy analysis registration schema")
    if set(raw) != {"schema", "split", "gsa_design"}:
        raise ValueError(f"{path}: registration must contain only schema, split, and gsa_design")
    split = _mapping("split", raw["split"])
    design = _mapping("gsa_design", raw["gsa_design"])
    if set(split) != {"payload", "payload_sha256"}:
        raise ValueError("split registration must contain payload and payload_sha256")
    if set(design) != {"payload", "design_sha256"}:
        raise ValueError("GSA registration must contain payload and design_sha256")
    return EnergyAnalysisRegistration(
        split_payload=_mapping("split.payload", split["payload"]),
        split_payload_sha256=split["payload_sha256"],
        gsa_design_payload=_mapping("gsa_design.payload", design["payload"]),
        design_sha256=design["design_sha256"],
    )


def load_energy_analysis_registration() -> EnergyAnalysisRegistration:
    """Load and hash-verify the repository's canonical energy registration."""
    return _load_energy_analysis_registration(_REGISTRATION_PATH)
