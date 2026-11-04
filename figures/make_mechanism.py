"""Figure: where the energy goes on the drift scenario (exploratory; Study F final grids).

Reads experiments/study_f_20261007/exploratory/MECHANISM.json; run from the repository root:
    .venv/bin/python -m experiments.study_f_20261007.exploratory.mechanism
    uv run python figures/make_mechanism.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
M = json.loads((ROOT / "experiments/study_f_20261007/exploratory/MECHANISM.json").read_text())
S = M["summary"]

# Categorical slots 1-3 of the validated reference palette (fixed order, never cycled).
DEVICE, CLOUD, NETWORK = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, SURFACE = "#1f1f1e", "#6b6b66", "#fcfcfb"

ROWS = [("graph_headroom", "Safe rule"),
        ("latency_greedy", "Energy-saving rule"),
        ("gp", "GP"),
        ("deepseek", "DeepSeek"),
        ("sol", "Sol")]


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, ax = plt.subplots(figsize=(3.4, 2.1), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (key, _name) in enumerate(ROWS):
        s = S[f"R|{key}"]
        y = len(ROWS) - 1 - i
        left = 0.0
        for value, color in ((s["mean_device_kj"], DEVICE), (s["mean_cloud_kj"], CLOUD),
                             (s["mean_network_kj"], NETWORK)):
            ax.barh(y, value, left=left, height=0.6, color=color, edgecolor=SURFACE,
                    linewidth=1.0)
            left += value
        ax.text(134, y, f"{s['feasible']}/{s['roots']}", va="center", ha="right",
                fontsize=6.5, color=INK if s["feasible"] == s["roots"] else MUTED)
    ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS][::-1], fontsize=7, color=INK)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", colors=MUTED, labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Mean modeled energy per 420 s episode (kJ)", fontsize=7, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (DEVICE, CLOUD, NETWORK)]
    ax.legend(handles, ["Fog/edge devices", "Cloud", "Network"], fontsize=6.5, frameon=False,
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, handlelength=1.0,
              columnspacing=1.0, borderaxespad=0.1)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "mechanism.pdf", facecolor=SURFACE)
    fig.savefig(HERE / "mechanism.png", facecolor=SURFACE)


if __name__ == "__main__":
    main()
