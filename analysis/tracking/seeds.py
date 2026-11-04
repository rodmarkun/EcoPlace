"""Deterministic seed derivation — the executable half of ``docs/SEEDS.md``.

Convention: one integer **root seed** per run (recorded in the run store), and every
stochastic component derives its own seed as ``derive_seed(root_seed, *component_path)``.
There are no default arguments anywhere: forgetting a seed is a ``TypeError`` at call time,
not a silent fall-back to wall-clock/OS entropy.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Derived seeds are confined to [0, 2**63) so they fit any consumer we use
# (numpy SeedSequence entropy, torch.manual_seed, Rust u64/i64, SQLite INTEGER).
_SEED_BITS = 63

MAX_ROOT_SEED = 2**63 - 1


def derive_seed(root_seed: int, *path: str | int) -> int:
    """Derive a child seed from ``root_seed`` and a non-empty component path.

    The derivation is a SHA-256 hash over an unambiguous encoding of
    ``(root_seed, *path)``, so it is stable across processes, platforms, and Python
    versions (unlike ``hash()``), and children of distinct paths are independent
    for all practical purposes.

    Args:
        root_seed: the run's root seed, ``0 <= root_seed < 2**63``.
        path: at least one path element identifying the consumer, e.g.
            ``derive_seed(root, "e1", "campaign", 3, "topology-gen")``.

    Returns:
        An integer in ``[0, 2**63)``.
    """
    if not isinstance(root_seed, int) or isinstance(root_seed, bool):
        raise TypeError(f"root_seed must be an int, got {type(root_seed).__name__}")
    if not 0 <= root_seed <= MAX_ROOT_SEED:
        raise ValueError(f"root_seed must be in [0, 2**63), got {root_seed}")
    if not path:
        raise ValueError("path must have at least one element (see docs/SEEDS.md)")

    h = hashlib.sha256()
    h.update(str(root_seed).encode("ascii"))
    for element in path:
        if isinstance(element, bool) or not isinstance(element, int | str):
            raise TypeError(
                f"path elements must be str or int, got {type(element).__name__}: {element!r}"
            )
        # Length-prefixed, type-tagged encoding: no separator collisions
        # (("ab","c") never collides with ("a","bc") or ("ab-c",)).
        tag = "i" if isinstance(element, int) else "s"
        data = str(element).encode("utf-8")
        h.update(f"|{tag}{len(data)}:".encode("ascii"))
        h.update(data)
    return int.from_bytes(h.digest()[:8], "big") >> (64 - _SEED_BITS)


def spawn_rng(root_seed: int, *path: str | int) -> np.random.Generator:
    """A numpy ``Generator`` seeded deterministically from ``(root_seed, *path)``.

    This is the only sanctioned way to obtain an RNG in this codebase; naked
    ``np.random.default_rng()`` (entropy-seeded) is banned by ``docs/SEEDS.md``.
    """
    return np.random.default_rng(derive_seed(root_seed, *path))
