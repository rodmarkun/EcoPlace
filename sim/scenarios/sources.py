"""Sealed Phase-4 workload-source split and provenance checks.

The split is orthogonal to scenario visibility.  Calibration APIs can resolve only
``calibration`` sources; G3 opens ``face-validity`` sources separately after P1.
The seal is mistake-proofing, while the registered file hashes are the durable audit
record.  Missing empirical payloads remain ``UNAVAILABLE``—metadata never becomes data.

A source whose evidence arrives as several distinct parser inputs (for example the
Alibaba v2021 metric member and call-graph member) registers one
:class:`PayloadComponent` per parser input.  A single ``payload_sha256`` cannot
authenticate two different parser inputs: registering either member's digest would
silently de-authorize the other, and a combined digest would authorize neither because
no consumer hashes the pair.  Component-aware sources therefore leave ``payload_sha256``
``None`` and every consumer binds to the exact component it parsed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "source_split.json"
PRODUCTION_MANIFEST_FILE_SHA256 = (
    "8e2ae68a78ea2d972d95fe67a2783e591009c7dfc1e99566e42d1db94d0cf578"
)
_AUTHORITY = object()
_SEAL_TAG = "workload-source-registry-v1:"
CALIBRATION = "calibration"
FACE_VALIDITY = "face-validity"
STRUCTURE = "structure"
_ROLES = (CALIBRATION, FACE_VALIDITY, STRUCTURE)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def clean_checkout_commit() -> str:
    """Return HEAD only when the tracked and untracked checkout is clean."""
    repo_root = Path(__file__).resolve().parents[2]
    checkout = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(checkout) != 40 or any(char not in "0123456789abcdef" for char in checkout):
        raise ValueError("repository HEAD must be a full SHA-1 object id")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("held-out access requires a clean frozen working tree")
    return checkout


def require_frozen_clean_checkout(frozen_commit: str) -> str:
    """Bind a held-out access gate to both HEAD and a clean tracked/untracked tree."""
    if len(frozen_commit) != 40 or any(
        char not in "0123456789abcdef" for char in frozen_commit
    ):
        raise ValueError("frozen commit must be a full SHA-1 object id")
    checkout = clean_checkout_commit()
    if checkout != frozen_commit:
        raise ValueError("current checkout does not match the frozen commit")
    return checkout


@dataclass(frozen=True, order=True)
class PayloadComponent:
    """One exact parser input belonging to a multi-part empirical source.

    ``sha256`` is the digest of the bytes a parser actually consumes (the extracted
    archive member), never the compressed archive; ``archive_sha256`` records the
    retained container so the acquisition can be re-verified independently.
    """

    component: str
    status: str
    parser_input: str
    sha256: str | None
    bytes: int | None
    archive_path: str | None
    archive_sha256: str | None
    archive_bytes: int | None
    notes: str

    def __post_init__(self) -> None:
        for name in ("component", "status", "parser_input", "notes"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"payload component {name} must be a non-empty string")
        if self.status not in ("AVAILABLE", "UNAVAILABLE"):
            raise ValueError("payload component status must be AVAILABLE or UNAVAILABLE")
        if self.status == "AVAILABLE":
            if self.sha256 is None or not _is_lower_hex(self.sha256, 64):
                raise ValueError("AVAILABLE payload component needs its parser-input SHA-256")
            if (
                isinstance(self.bytes, bool)
                or not isinstance(self.bytes, int)
                or self.bytes <= 0
            ):
                raise ValueError("AVAILABLE payload component needs a positive byte count")
        elif self.sha256 is not None or self.bytes is not None:
            raise ValueError("an unavailable payload component may not advertise bytes or a hash")
        if (self.archive_path is None) != (self.archive_sha256 is None):
            raise ValueError("archive_path and archive_sha256 must appear together")
        if self.archive_sha256 is not None and not _is_lower_hex(self.archive_sha256, 64):
            raise ValueError("archive_sha256 must be a lowercase SHA-256 digest")
        if (self.archive_path is None) != (self.archive_bytes is None):
            raise ValueError("archive_path and archive_bytes must appear together")
        if self.archive_bytes is not None and (
            isinstance(self.archive_bytes, bool)
            or not isinstance(self.archive_bytes, int)
            or self.archive_bytes <= 0
        ):
            raise ValueError("archive_bytes must be a positive integer")


@dataclass(frozen=True)
class SourceArtifact:
    source_id: str
    role: str
    kind: str
    version: str
    repo_commit: str | None
    url: str
    retained_path: str | None
    retained_sha256: str | None
    capabilities: tuple[str, ...]
    payload_status: str
    payload_sha256: str | None
    payload_components: tuple[PayloadComponent, ...]
    notes: str

    def component(self, name: str) -> PayloadComponent:
        """Return the named registered parser input, or raise ``KeyError``."""
        try:
            return next(item for item in self.payload_components if item.component == name)
        except StopIteration as exc:
            raise KeyError(f"{self.source_id}:{name}") from exc

    def authorized_digests(self) -> tuple[str, ...]:
        """Every exact parser-input digest this registration authorizes."""
        if self.payload_components:
            return tuple(
                item.sha256
                for item in self.payload_components
                if item.status == "AVAILABLE" and item.sha256 is not None
            )
        if self.payload_status == "AVAILABLE" and self.payload_sha256 is not None:
            return (self.payload_sha256,)
        return ()

    def __post_init__(self) -> None:
        for name in ("source_id", "kind", "version", "url", "payload_status", "notes"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"source {name} must be a non-empty string")
        if self.role not in _ROLES:
            raise ValueError(f"source role must be one of {_ROLES}")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        if not self.capabilities or len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("source capabilities must be non-empty and unique")
        if (self.retained_path is None) != (self.retained_sha256 is None):
            raise ValueError("retained_path and retained_sha256 must appear together")
        if self.retained_sha256 is not None and not _is_lower_hex(self.retained_sha256, 64):
            raise ValueError("retained_sha256 must be a lowercase SHA-256 digest")
        if self.repo_commit is not None and not _is_lower_hex(self.repo_commit, 40):
            raise ValueError("repo_commit must be a full lowercase Git object id")
        object.__setattr__(self, "payload_components", tuple(self.payload_components))
        if any(not isinstance(item, PayloadComponent) for item in self.payload_components):
            raise ValueError("payload_components must be typed PayloadComponent records")
        names = [item.component for item in self.payload_components]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("payload component names must be unique and canonically sorted")
        if self.payload_components:
            # One digest cannot authenticate several distinct parser inputs.
            if self.payload_sha256 is not None:
                raise ValueError(
                    "a component-aware source must not also advertise a single payload hash"
                )
            available = all(item.status == "AVAILABLE" for item in self.payload_components)
            expected = "AVAILABLE" if available else "UNAVAILABLE"
            if self.payload_status != expected:
                raise ValueError(
                    "component-aware payload_status must be AVAILABLE only when every "
                    "registered component is AVAILABLE"
                )
            return
        if self.payload_status == "AVAILABLE":
            if self.payload_sha256 is None or not _is_lower_hex(self.payload_sha256, 64):
                raise ValueError("AVAILABLE source payload needs its SHA-256")
        elif self.payload_status != "UNAVAILABLE":
            raise ValueError("payload_status must be AVAILABLE or UNAVAILABLE")
        elif self.payload_sha256 is not None:
            raise ValueError("an unavailable source may not advertise a payload hash")


@dataclass(frozen=True)
class SourceRegistry:
    schema: str
    declared_before_fitting: bool
    sources: tuple[SourceArtifact, ...]
    seal: str
    _authority: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _AUTHORITY:
            raise ValueError("SourceRegistry must be obtained from load_source_registry()")
        if not self.declared_before_fitting:
            raise ValueError("source split must be declared before fitting")
        if self.schema != "workload-source-split-v1":
            raise ValueError("unknown workload source registry schema")
        ids = [source.source_id for source in self.sources]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("source ids must be unique and canonically sorted")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "declared_before_fitting": self.declared_before_fitting,
            "sources": [dataclasses.asdict(source) for source in self.sources],
        }

    def verify_seal(self) -> None:
        expected = hashlib.sha256((_SEAL_TAG + _canonical(self._payload())).encode()).hexdigest()
        if self.seal != expected:
            raise ValueError("workload source registry seal mismatch")

    def get(self, source_id: str) -> SourceArtifact:
        self.verify_seal()
        try:
            return next(source for source in self.sources if source.source_id == source_id)
        except StopIteration as exc:
            raise KeyError(source_id) from exc

    def require_role(self, source_id: str, role: str) -> SourceArtifact:
        source = self.get(source_id)
        if source.role != role:
            raise ValueError(
                f"source {source_id!r} is {source.role!r}, not permitted role {role!r}"
            )
        return source

    def require_payload(self, source_id: str, role: str, content_sha256: str) -> SourceArtifact:
        """Resolve a role and bind the consumer to some exact registered parser input.

        Component-aware sources authorize each registered component's own digest.  Use
        :meth:`require_component` wherever the consumer knows which parser input it read;
        that binding additionally prevents one component's bytes from being accepted in
        another component's role.
        """
        source = self.require_role(source_id, role)
        digests = source.authorized_digests()
        if source.payload_status != "AVAILABLE" or not digests:
            raise ValueError(f"registered empirical payload for {source_id!r} is unavailable")
        if content_sha256 not in digests:
            raise ValueError(f"payload hash mismatch for source {source_id!r}")
        return source

    def require_component(
        self, source_id: str, role: str, component: str, content_sha256: str
    ) -> SourceArtifact:
        """Bind a consumer to one exact named parser input of a registered source.

        Sources registered under the single-payload contract (no declared components)
        fall back to :meth:`require_payload`; the component name is then advisory only.
        """
        source = self.require_role(source_id, role)
        if not source.payload_components:
            return self.require_payload(source_id, role, content_sha256)
        try:
            record = source.component(component)
        except KeyError as exc:
            raise ValueError(
                f"source {source_id!r} registers no payload component {component!r}"
            ) from exc
        if record.status != "AVAILABLE" or record.sha256 is None:
            raise ValueError(
                f"registered payload component {source_id!r}:{component!r} is unavailable"
            )
        if record.sha256 != content_sha256:
            raise ValueError(
                f"payload hash mismatch for source component {source_id!r}:{component!r}"
            )
        return source

    def verify_payload_components(self, workspace: Path) -> None:
        """Re-verify every retained component archive against its registered digest.

        Deliberately separate from :meth:`verify_retained`: component archives are
        gigabyte-scale acquisition containers, so this runs in ingestion drivers rather
        than in the fast metadata tests.
        """
        self.verify_seal()
        root = workspace.resolve()
        for source in self.sources:
            for item in source.payload_components:
                if item.status != "AVAILABLE" or item.archive_path is None:
                    continue
                path = (root / item.archive_path).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f"retained payload archive missing/escaping: {path}")
                if item.archive_bytes is not None and path.stat().st_size != item.archive_bytes:
                    raise ValueError(
                        f"payload archive byte count mismatch for "
                        f"{source.source_id}:{item.component}"
                    )
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 22), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item.archive_sha256:
                    raise ValueError(
                        f"payload archive hash mismatch for "
                        f"{source.source_id}:{item.component}: {digest.hexdigest()}"
                    )

    def verify_retained(self, workspace: Path) -> None:
        self.verify_seal()
        root = workspace.resolve()
        for source in self.sources:
            if source.retained_path is None:
                continue
            path = (root / source.retained_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"retained source artifact missing/escaping: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != source.retained_sha256:
                raise ValueError(f"retained source hash mismatch for {source.source_id}: {actual}")

    def manifest_sha256(self) -> str:
        self.verify_seal()
        return hashlib.sha256(_canonical(self._payload()).encode()).hexdigest()


def _load_source_registry_bytes(payload: bytes) -> SourceRegistry:
    raw = json.loads(payload)
    allowed = {"schema", "declared_before_fitting", "sources"}
    if set(raw) != allowed:
        raise ValueError(f"source registry keys must be exactly {sorted(allowed)}")
    sources = tuple(
        sorted(
            (
                SourceArtifact(
                    **{
                        **entry,
                        "capabilities": tuple(entry["capabilities"]),
                        "payload_components": tuple(
                            PayloadComponent(**item)
                            for item in entry["payload_components"]
                        ),
                    }
                )
                for entry in raw["sources"]
            ),
            key=lambda source: source.source_id,
        )
    )
    payload = {
        "schema": raw["schema"],
        "declared_before_fitting": raw["declared_before_fitting"],
        "sources": [dataclasses.asdict(source) for source in sources],
    }
    seal = hashlib.sha256((_SEAL_TAG + _canonical(payload)).encode()).hexdigest()
    return SourceRegistry(
        schema=raw["schema"],
        declared_before_fitting=raw["declared_before_fitting"],
        sources=sources,
        seal=seal,
        _authority=_AUTHORITY,
    )


def load_source_registry() -> SourceRegistry:
    """Load only the repository-pinned production split, never a caller path."""
    payload = DATA_PATH.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != PRODUCTION_MANIFEST_FILE_SHA256:
        raise ValueError(
            "production source split bytes changed without updating the pinned manifest hash"
        )
    return _load_source_registry_bytes(payload)
