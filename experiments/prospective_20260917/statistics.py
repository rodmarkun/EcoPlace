"""Prospective root-level analysis. Never treats episodes as independent searches."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest

NUMERICAL_TIE = 1e-9
PRACTICAL_ENERGY = 0.01
BOOTSTRAP_RESAMPLES = 20000


def direction(left: dict, right: dict, *, tolerance: float = NUMERICAL_TIE) -> int | None:
    for row in (left, right):
        if row["status"] == "missing":
            return None
        if row["status"] != "evaluated" or type(row["feasible"]) is not bool:
            raise ValueError("unknown result state; faults require explicit missing status")
        if row["feasible"] and (row.get("energy") is None or not math.isfinite(row["energy"])):
            raise ValueError("feasible outcome requires finite energy")
    if left["feasible"] != right["feasible"]:
        return 1 if left["feasible"] else -1
    if not left["feasible"]:
        return 0
    delta = right["energy"] - left["energy"]
    return 0 if abs(delta) <= tolerance else (1 if delta > 0 else -1)


def sign_summary(directions: list[int | None]) -> dict:
    if any(d not in (-1, 0, 1, None) for d in directions):
        raise ValueError("invalid direction")
    w, losses, t, missing = (directions.count(d) for d in (1, -1, 0, None))
    result = {
        "wins": w,
        "losses": losses,
        "ties": t,
        "missing": missing,
        "p_two_sided": None,
        "conditional_win_probability_ci95": None,
    }
    if missing:
        return result
    result["p_two_sided"] = 1.0
    if w + losses:
        test = binomtest(w, w + losses, 0.5, alternative="two-sided")
        ci = test.proportion_ci(confidence_level=0.95, method="exact")
        result.update(
            p_two_sided=float(test.pvalue),
            conditional_win_probability_ci95=[float(ci.low), float(ci.high)],
        )
    return result


def holm_two(pvalues: list[float | None]) -> list[float | None]:
    if len(pvalues) != 2 or any(p is not None and not 0 <= p <= 1 for p in pvalues):
        raise ValueError("exactly two valid primary p-values required")
    # Incomplete family cannot support a confirmatory claim.
    if None in pvalues:
        return [None, None]
    order = sorted(range(2), key=lambda i: (pvalues[i], i))
    adjusted = [0.0, 0.0]
    adjusted[order[0]] = min(1.0, 2 * pvalues[order[0]])
    adjusted[order[1]] = max(adjusted[order[0]], pvalues[order[1]])
    return adjusted


def paired_report(rows: list[dict], roots: list[int], *, bootstrap_seed: int) -> dict:
    by_root = {r["root_seed"]: r for r in rows}
    if len(rows) != len(by_root) or set(by_root) != set(roots) or len(set(roots)) != len(roots):
        raise ValueError("complete unique registered root population required")
    ordered = [by_root[root] for root in roots]
    dirs = [direction(r["left"], r["right"]) for r in ordered]
    practical = [direction(r["left"], r["right"], tolerance=PRACTICAL_ENERGY) for r in ordered]
    deltas = [
        r["right"]["energy"] - r["left"]["energy"]
        for r in ordered
        if r["left"]["status"] == r["right"]["status"] == "evaluated"
        and r["left"]["feasible"]
        and r["right"]["feasible"]
    ]
    energy = {
        "jointly_feasible": len(deltas),
        "mean": None,
        "median": None,
        "bootstrap_ci95_descriptive": None,
    }
    if deltas:
        a = np.asarray(deltas)
        energy.update(mean=float(a.mean()), median=float(np.median(a)))
        if len(a) >= 2:
            rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
            means = a[rng.integers(0, len(a), size=(BOOTSTRAP_RESAMPLES, len(a)))].mean(axis=1)
            energy["bootstrap_ci95_descriptive"] = np.quantile(
                means, [0.025, 0.975], method="linear"
            ).tolist()
    return {
        "primary": sign_summary(dirs),
        "practical_directions": practical,
        "energy": energy,
        "root_directions": dict(zip(roots, dirs, strict=True)),
        "feasibility": {
            side: sum(r[side]["status"] == "evaluated" and r[side]["feasible"] for r in ordered)
            for side in ("left", "right")
        },
    }


def power(n: int, win: float, ties: float, alpha: float) -> float:
    return sum(
        math.comb(n, m)
        * (1 - ties) ** m
        * ties ** (n - m)
        * sum(
            math.comb(m, k) * win**k * (1 - win) ** (m - k)
            for k in range(m + 1)
            if min(1.0, 2 * sum(math.comb(m, j) for j in range(min(k, m - k) + 1)) / 2**m) <= alpha
        )
        for m in range(n + 1)
    )
