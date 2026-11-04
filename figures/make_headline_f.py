"""Figure: Sol's winning policies against each competitor on fresh final seeds (90 runs).

Reads only the committed analysis; run from the repository root:
    uv run python figures/make_headline_f.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
F = json.loads((ROOT / "experiments/study_f_20261007/analysis/PRIMARY.json").read_text())

BLUE, ORANGE, TIE = "#2a78d6", "#eb6834", "#d9d9d4"
INK, MUTED = "#1f1f1e", "#6b6b66"


def row(name, block):
    return name, (block["wins"], block["ties"], block["losses"]), (
        block["mean_relative_saving"], block["jointly_feasible"])


def pooled():
    c, d = F["confirmatory"], F["pooled_descriptive"]["contrasts"]
    return [row("DeepSeek", c["S1 sol_validated vs deepseek_validated"]),
            row("GP", c["P sol_validated vs gp_validated"]),
            row("best hand-written rule", d["sol_validated vs oracle_reference"]),
            row("best starting program", d["sol_validated vs initialization_best"])]


def families():
    rows = []
    for key, short in (("sol_validated vs deepseek_validated", "DeepSeek"),
                       ("sol_validated vs gp_validated", "GP")):
        for fam, label in (("T", "baseline"), ("W", "wider"), ("R", "reduced")):
            rows.append(row(f"{short}, {label}", F["per_family"][fam]["contrasts"][key]))
    return rows


def draw(ax, rows, total):
    for i, (_name, (w, t, l), (sav, n)) in enumerate(rows):  # noqa: E741
        y = len(rows) - 1 - i
        for start, width, color, text_color in ((0, w, BLUE, "white"), (w, t, TIE, MUTED),
                                                (w + t, l, ORANGE, "white")):
            if width:
                ax.barh(y, width, left=start, height=0.62, color=color, edgecolor="white",
                        linewidth=1.2)
                if width >= total / 15:
                    ax.text(start + width / 2, y, str(width), ha="center", va="center",
                            fontsize=6.5, color=text_color)
        ax.text(total * 1.03, y, f"{100 * sav:.1f}% ({n})", va="center", fontsize=6.5,
                color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=7, color=INK)
    ax.set_xlim(0, total)
    ax.set_xticks([0, total / 3, 2 * total / 3, total])
    ax.tick_params(axis="x", colors=MUTED, labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.text(total * 1.03, len(rows) - 0.35, "Sol's saving\nwhen both met\ntargets (runs)",
            fontsize=6.5, color=MUTED, va="bottom")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 4.2), dpi=300,
                                      gridspec_kw={"height_ratios": [4, 6]})
    draw(top, pooled(), 90)
    top.set_xlabel("Runs (of 90)", fontsize=7, color=INK)
    top.set_title("All runs: Sol's policy against", fontsize=7.5, color=INK, loc="left",
                  x=-0.9, pad=26)
    draw(bottom, families(), 30)
    bottom.set_xlabel("Runs (of 30 per scenario)", fontsize=7, color=INK)
    bottom.set_title("By scenario: Sol's policy against", fontsize=7.5, color=INK, loc="left",
                     x=-0.9, pad=26)
    fig.subplots_adjust(left=0.36, right=0.75, top=0.86, bottom=0.13, hspace=0.95)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, TIE, ORANGE)]
    fig.legend(handles, ["Sol wins", "tie", "Sol loses"], loc="lower left", ncol=3,
               bbox_to_anchor=(0.30, 0.0), frameon=False, fontsize=6.5, handlelength=1.0)
    fig.savefig(HERE / "headline_f.pdf")
    fig.savefig(HERE / "headline_f.png")
    for r in pooled() + families():
        print(r)


if __name__ == "__main__":
    main()
