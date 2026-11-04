"""Apply the criteria fixed in artifacts/PLAN.json to the ladder fidelity results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / "artifacts"
TOP_K = 5
MIN_FEASIBLE = 10
MAX_FALSE_ELIMINATION = 0.05


def ranks(keys):
    """Average ranks (1 = best) of lexicographic sort keys, ties sharing their mean rank."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    out = [0.0] * len(keys)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else None


def summarize(study, rows, expected):
    errors = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]
    feasible = [r for r in ok if r["l2"]["feasible"]]
    eliminated = [r for r in feasible if not r["l1_promote"]]
    top = sorted(ok, key=lambda r: r["l2"]["sort_key"])[:TOP_K]
    f1_evaluable = len(feasible) >= MIN_FEASIBLE
    f1_rate = len(eliminated) / len(feasible) if feasible else None
    f1 = (f1_rate <= MAX_FALSE_ELIMINATION) if f1_evaluable else None
    f2 = all(r["l1_promote"] for r in top)
    return {
        "study": study,
        "policies": len(rows),
        "expected": expected,
        "complete": len(rows) == expected,
        "evaluation_errors": len(errors),
        "l0_valid": sum(r["l0_valid"] for r in ok),
        "l1_promoted": sum(r["l1_promote"] for r in ok),
        "l1_reject_rate": 1 - sum(r["l1_promote"] for r in ok) / len(ok),
        "l2_feasible": len(feasible),
        "F1_false_elimination": {
            "evaluable": f1_evaluable,
            "l2_feasible_rejected_by_l0_or_l1": len(eliminated),
            "rate": f1_rate,
            "threshold": MAX_FALSE_ELIMINATION,
            "pass": f1,
            "eliminated": [
                {"index": r["index"], "path": r["path"], "l1": r["l1"], "l2": r["l2"]}
                for r in eliminated
            ],
        },
        "F2_top_k_retention": {
            "k": TOP_K,
            "pass": f2,
            "top": [
                {"index": r["index"], "stratum": r["stratum"], "l1_promote": r["l1_promote"],
                 "l2_feasible": r["l2"]["feasible"], "l2_sort_key": r["l2"]["sort_key"]}
                for r in top
            ],
        },
        "F3_descriptive": {
            "spearman_l1_l2_all": spearman(
                [r["l1"]["sort_key"] for r in ok], [r["l2"]["sort_key"] for r in ok]
            ),
            "l2_feasibility_rate": len(feasible) / len(ok),
            "by_stratum": {
                s: {
                    "n": sum(r["stratum"] == s for r in ok),
                    "l1_promoted": sum(r["stratum"] == s and r["l1_promote"] for r in ok),
                    "l2_feasible": sum(r["stratum"] == s and r["l2"]["feasible"] for r in ok),
                }
                for s in sorted({r["stratum"] for r in ok})
            },
        },
        # Registered rule (PLAN.json "decision"): F1 (if evaluable) and F2. Policies that
        # trap at runtime are invalid at every rung and reported separately; they cannot be
        # L2-feasible (see TRAP_L2_CHECK.json).
        "ladder_pass": bool(f2 and f1 is not False),
        "runtime_trap_policies": [
            {"index": r["index"], "path": r["path"], "error": r["error"][:200]} for r in errors
        ],
        "note": None if f1_evaluable else "F1 underpowered (<10 L2-feasible); not a pass of F1.",
    }


def main(studies):
    plan = json.loads((OUT / "PLAN.json").read_text())
    rows = [json.loads(line) for line in (OUT / "results.jsonl").open()]
    result = {
        s: summarize(s, [r for r in rows if r["study"] == s], len(plan["corpus"][s]))
        for s in studies
    }
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    studies = sys.argv[1:] or ["A", "B"]
    result = main(studies)
    if studies == ["A", "B"] and all(r["complete"] for r in result.values()):
        with (OUT / "SUMMARY.json").open("x") as stream:
            json.dump(result, stream, indent=1, sort_keys=True)
            stream.write("\n")
