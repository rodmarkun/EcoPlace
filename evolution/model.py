"""Single open-weight model pin, host handshake, and hard API-cost ledger (TODO 5.7)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_MANIFEST_SCHEMA = "evolved-continuum-open-model-v2"


def _digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def host_handshake_sha256(handshake: dict[str, Any]) -> str:
    """Digest the semantic handshake, independent of HTTP JSON whitespace."""
    return _digest(handshake)


@dataclass(frozen=True)
class WeightFile:
    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.path.endswith(".safetensors") or Path(self.path).name != self.path:
            raise ValueError("weight paths must be top-level safetensors filenames")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise ValueError("weight size must be an int")
        if self.size_bytes <= 0 or len(self.sha256) != 64:
            raise ValueError("weight files require positive size and sha256")

    def payload(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class InferenceHost:
    status: str
    accelerator: str
    accelerator_memory_gb: int
    cpu_arch: str
    os: str
    engine: str
    engine_version: str
    transformers_version: str
    dtype: str
    tensor_parallel_size: int
    max_model_len: int
    isolation: str
    request_idempotency: str
    cost_enforcement: str
    max_request_cost_microusd: int
    weight_verification: str

    def __post_init__(self) -> None:
        if self.status not in {"development-required-profile", "p2-frozen"}:
            raise ValueError("unknown inference-host status")
        if self.engine != "vllm-openai-compatible" or self.dtype != "bfloat16":
            raise ValueError("the model track pins the vLLM BF16 serving path")
        if self.tensor_parallel_size != 1:
            raise ValueError("the pinned single-H100 host uses tensor_parallel_size=1")
        if "proposal" not in self.isolation or "evaluator" not in self.isolation:
            raise ValueError("host isolation must explicitly separate proposal and evaluator paths")
        if self.request_idempotency != "model-and-seed-response-cache-required":
            raise ValueError("proposal host must make model+seed request ids idempotent")
        if self.cost_enforcement != "reject-before-generation-above-max-request-cost":
            raise ValueError("proposal host must enforce the cost cap before generation")
        if (
            not isinstance(self.max_request_cost_microusd, int)
            or isinstance(self.max_request_cost_microusd, bool)
            or self.max_request_cost_microusd < 1
        ):
            raise ValueError("host max request cost must be a positive integer")
        if self.weight_verification != "startup-size-and-sha256-scan-required":
            raise ValueError("proposal host must verify every pinned weight at startup")


@dataclass(frozen=True)
class ModelTrack:
    track_id: str
    model_id: str
    revision: str
    license: str
    source_url: str
    accessed_date: str
    weights: tuple[WeightFile, ...]
    weight_set_sha256: str
    generation: dict[str, Any]
    host: InferenceHost
    max_api_costs_microusd: int
    max_call_reservation_microusd: int

    def __post_init__(self) -> None:
        if self.track_id != "open-weight-headline" or self.license != "apache-2.0":
            raise ValueError("Phase 5 permits exactly one Apache-2.0 open-weight track")
        if len(self.revision) != 40 or len(self.weight_set_sha256) != 64:
            raise ValueError("model revision and weight-set digest must be pinned")
        paths = [weight.path for weight in self.weights]
        if not self.weights or len(paths) != len(set(paths)):
            raise ValueError("weight manifest must be non-empty and duplicate-free")
        if self.weight_set_sha256 != _digest([weight.payload() for weight in self.weights]):
            raise ValueError("weight_set_sha256 does not cover the ordered weight manifest")
        for field in ("max_api_costs_microusd", "max_call_reservation_microusd"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.max_call_reservation_microusd > self.max_api_costs_microusd:
            raise ValueError("per-call cost reservation exceeds total max_api_costs")
        if self.host.max_request_cost_microusd != self.max_call_reservation_microusd:
            raise ValueError("host and campaign per-request cost caps differ")
        expected_generation = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_new_tokens": 4096,
        }
        if self.generation != expected_generation:
            raise ValueError("generation settings must equal the registered Qwen track")

    @property
    def confirmatory_ready(self) -> bool:
        return self.host.status == "p2-frozen"


@dataclass
class ApiCostLedger:
    """Integer-microdollar ledger; reserve before dispatch, charge after response."""

    cap_microusd: int
    spent_microusd: int = 0
    reserved_microusd: int = 0

    def __post_init__(self) -> None:
        for field in ("cap_microusd", "spent_microusd", "reserved_microusd"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        self._check()

    def _check(self) -> None:
        if self.spent_microusd + self.reserved_microusd > self.cap_microusd:
            raise ValueError("max_api_costs exceeded")

    def reserve(self, amount_microusd: int) -> None:
        if not isinstance(amount_microusd, int) or isinstance(amount_microusd, bool):
            raise TypeError("cost reservations use integer microdollars")
        if amount_microusd < 0:
            raise ValueError("cost reservation must be non-negative")
        self.reserved_microusd += amount_microusd
        try:
            self._check()
        except ValueError:
            self.reserved_microusd -= amount_microusd
            raise

    def settle(self, reserved_microusd: int, actual_microusd: int) -> None:
        for value in (reserved_microusd, actual_microusd):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("cost settlement uses non-negative integer microdollars")
        if reserved_microusd > self.reserved_microusd:
            raise ValueError("cannot settle more than the outstanding reservation")
        if actual_microusd > reserved_microusd:
            raise ValueError("actual cost exceeds its hard pre-dispatch reservation")
        self.reserved_microusd -= reserved_microusd
        self.spent_microusd += actual_microusd
        self._check()

    def to_dict(self) -> dict[str, int]:
        return {
            "cap_microusd": self.cap_microusd,
            "spent_microusd": self.spent_microusd,
            "reserved_microusd": self.reserved_microusd,
            "remaining_microusd": self.cap_microusd - self.spent_microusd - self.reserved_microusd,
        }


def load_model_track(path: str | Path) -> ModelTrack:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise ValueError(f"model manifest schema must be {MODEL_MANIFEST_SCHEMA}")
    tracks = raw.get("tracks")
    if not isinstance(tracks, list) or len(tracks) != 1:
        raise ValueError("exactly one model track is permitted")
    item = tracks[0]
    return ModelTrack(
        track_id=item["track_id"],
        model_id=item["model_id"],
        revision=item["revision"],
        license=item["license"],
        source_url=item["source_url"],
        accessed_date=item["accessed_date"],
        weights=tuple(WeightFile(**weight) for weight in item["weights"]),
        weight_set_sha256=item["weight_set_sha256"],
        generation=item["generation"],
        host=InferenceHost(**item["inference_host"]),
        max_api_costs_microusd=item["max_api_costs_microusd"],
        max_call_reservation_microusd=item["max_call_reservation_microusd"],
    )


def verify_weight_directory(track: ModelTrack, directory: str | Path) -> None:
    root = Path(directory)
    for weight in track.weights:
        path = root / weight.path
        if not path.is_file() or path.stat().st_size != weight.size_bytes:
            raise ValueError(f"weight file missing or wrong size: {weight.path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != weight.sha256:
            raise ValueError(f"weight sha256 mismatch: {weight.path}")


def verify_host_handshake(track: ModelTrack, handshake: dict[str, Any]) -> None:
    """Fail closed unless the isolated proposal server reports every pinned identity."""
    expected = {
        "model_id": track.model_id,
        "revision": track.revision,
        "weight_set_sha256": track.weight_set_sha256,
        "accelerator": track.host.accelerator,
        "accelerator_memory_gb": track.host.accelerator_memory_gb,
        "cpu_arch": track.host.cpu_arch,
        "os": track.host.os,
        "engine": track.host.engine,
        "engine_version": track.host.engine_version,
        "transformers_version": track.host.transformers_version,
        "dtype": track.host.dtype,
        "tensor_parallel_size": track.host.tensor_parallel_size,
        "max_model_len": track.host.max_model_len,
        "isolation": track.host.isolation,
        "request_idempotency": track.host.request_idempotency,
        "cost_enforcement": track.host.cost_enforcement,
        "max_request_cost_microusd": track.host.max_request_cost_microusd,
        "weight_verification": track.host.weight_verification,
    }
    if handshake != expected:
        different = sorted(
            key for key in set(expected) | set(handshake) if expected.get(key) != handshake.get(key)
        )
        raise ValueError(f"inference host handshake differs from pin: {different}")
