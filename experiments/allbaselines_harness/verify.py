"""Constant-memory verification for sealed harness JSON and legacy ``.json.gz``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .support import canonical_json, verify_sealed_stream


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify sealed harness JSON. A missing FILE.json is resolved as FILE.json.gz "
            "without decompressing it onto disk."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args()


def _sealed_paths(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    paths = []
    for name in ("report.json", "preflight.json", "SYNTHESIS.json"):
        candidate = path / name
        if candidate.exists() or candidate.with_name(f"{candidate.name}.gz").exists():
            paths.append(candidate)
    paths.extend(sorted(path.glob("inputs/*manifest.json")))
    paths.extend(sorted(path.glob("arms/*/result.json")))
    paths.extend(
        compressed.with_suffix("")
        for compressed in sorted(path.glob("arms/*/result.json.gz"))
        if not compressed.with_suffix("").exists()
    )
    if not paths:
        raise ValueError(f"directory contains no recognized sealed artifacts: {path}")
    return paths


def verify(paths: list[Path]) -> dict[str, Any]:
    artifacts = []
    seen = set()
    for requested in paths:
        for path in _sealed_paths(requested):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            artifacts.append(verify_sealed_stream(path))
    return {
        "status": "VERIFIED",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> int:
    args = _arguments()
    print(canonical_json(verify(args.paths)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
