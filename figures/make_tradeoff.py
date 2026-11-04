"""Figure: robustness-energy trade-off by search objective (final-grid outcomes).

Study A (one search seed) with the exploratory validation re-selection, study C (four search
seeds) and, once analysed, study D (C's roots with one search seed).

Reads only committed result files; run from the repository root:
    uv run python figures/make_tradeoff.py
"""

import json
import statistics as S
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
PRIMARY = json.loads((ROOT / "experiments/corrected_ab_20260925/analysis/PRIMARY.json")
                     .read_text())["studies"]["A"]
STUDY_C = json.loads((ROOT / "experiments/study_c_20261003/analysis/PRIMARY.json").read_text())
STUDY_D_PATH = ROOT / "experiments/study_d_20261004/analysis/PRIMARY.json"
BRITTLE = json.loads((ROOT / "experiments/brittleness_20261002/SUMMARY.json").read_text())

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, MUTED = "#1f1f1e", "#6b6b66"


def point(label):
    rows = list(PRIMARY["outcomes"].values())
    feas = [r[label] for r in rows if r[label]["feasible"]]
    return len(feas) / len(rows), S.mean(r["energy"] for r in feas)


def point_c(label):
    return STUDY_C["feasibility_frequency"][label], STUDY_C["mean_energy_when_feasible"][label]


def point_d(label):
    d = json.loads(STUDY_D_PATH.read_text())
    return (d["feasibility_frequency"][f"{label}@D"],
            d["mean_energy_when_feasible"][f"{label}@D"])


def validated(arm):
    v = BRITTLE["brittleness"]["A"][arm]["validated_champion"]
    return v["feasible_fraction"], v["mean_energy_when_feasible"]


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
    ax.grid(True, color="#e6e6e3", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    marks = [
        ("llm_champion", "LLM, search champion", BLUE, "o"),
        ("gp_champion", "GP, search champion", ORANGE, "s"),
        ("initialization_best", "Initialization best", AQUA, "D"),
        ("reference:latency_greedy", "latency_greedy / network_overhead", GRAY, "^"),
    ]
    offsets = {"llm_champion": (7, -4), "gp_champion": (0, -9),
               "initialization_best": (-4, 8), "reference:latency_greedy": (0, 10)}
    names = {"reference:latency_greedy": "latency_greedy\n& network_overhead",
             "llm_champion": "LLM, 1-seed search (A)", "gp_champion": "GP, 1-seed (A)",
             "initialization_best": "Init. best (A)"}
    for label, text, color, marker in marks:
        x, y = point(label)
        ax.scatter([x], [y], s=46, color=color, marker=marker, edgecolor="white",
                   linewidth=1.2, zorder=3)
        dx, dy = offsets[label]
        ax.annotate(names.get(label, text), (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK,
                    ha="right" if dx < 0 else ("center" if dx == 0 else "left"), va="center")

    for arm, color, marker in (("llm", BLUE, "o"), ("gp", ORANGE, "s")):
        x0, y0 = point(f"{arm}_champion")
        x1, y1 = validated(arm)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4,
                                "shrinkA": 5, "shrinkB": 5})
        ax.scatter([x1], [y1], s=46, facecolor="white", edgecolor=color, marker=marker,
                   linewidth=1.6, zorder=3)
        ax.annotate(f"{arm.upper()}, A re-selected", (x1, y1),
                    xytext=(0, 9) if arm == "gp" else (6, -9), textcoords="offset points",
                    fontsize=6.5, color=INK, va="center", ha="right" if arm == "gp" else "left")

    # Study C (filled stars) and, when available, study D on the same roots (open stars).
    for arm, color in (("llm", BLUE), ("gp", ORANGE)):
        xc, yc = point_c(f"{arm}_champion")
        ax.scatter([xc], [yc], s=70, color=color, marker="*", edgecolor="white", linewidth=0.8,
                   zorder=4)
        ax.annotate(f"{arm.upper()}, 4-seed (C)", (xc, yc), xytext=(0, 9),
                    textcoords="offset points", fontsize=6.5, color=INK, ha="center",
                    va="center")
        if STUDY_D_PATH.exists():
            xd, yd = point_d(f"{arm}_champion")
            ax.plot([xd, xc], [yd, yc], color=color, lw=0.8, ls=(0, (2, 2)), zorder=2)
            ax.scatter([xd], [yd], s=70, facecolor="white", edgecolor=color, marker="*",
                       linewidth=1.0, zorder=4)
            ax.annotate(f"{arm.upper()}, 1-seed (D)", (xd, yd),
                        xytext=(-10, 2) if arm == "gp" else (6, 0), textcoords="offset points",
                        fontsize=6.5, color=INK, va="center",
                        ha="right" if arm == "gp" else "left")

    ax.set_xlim(-0.04, 1.08)
    ax.set_xlabel("Feasible on fresh final seeds (fraction of roots)")
    ax.set_ylabel("Mean normalized energy\nwhen feasible (lower = better)")
    ax.set_ylim(0.62, 0.88)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "tradeoff.pdf")
    fig.savefig(HERE / "tradeoff.png")


if __name__ == "__main__":
    main()
