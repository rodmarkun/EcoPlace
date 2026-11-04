"""Figure: the clusters of the baseline and wider-network scenarios (reduced capacity = baseline
with three nodes closed). Reads the scenario files; run from the repository root:
    uv run python figures/make_topology.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
SCEN = ROOT / "sim/scenarios/dev"
CLOSED = {0, 2, 10}  # node_churn of continuum_drift.json (reduced capacity), 150-330 s

INK, MUTED, LINE = "#1f1f1e", "#6b6b66", "#b9b9b3"
CLOUD, FOG, EDGE, CLOSE = "#6d8fbf", "#2a78d6", "#9cc3ee", "#eb6834"


def sites(path):
    c = json.loads(path.read_text())["cluster"]
    lat, nodes = c["latency_ms"], c["nodes"]
    fogs = [n["id"] for n in nodes if n["tier"] == 1]
    edges = [n["id"] for n in nodes if n["tier"] == 2]
    groups, seen = [], set()
    for f in fogs:
        members = [e for e in edges if lat[e][f] <= 5.0]
        groups.append((f, members))
        seen.update(members)
    rest = [e for e in edges if e not in seen]
    while rest:  # fog-less sites: devices 2 ms apart
        head = rest[0]
        members = [e for e in rest if e == head or lat[head][e] <= 2.0]
        groups.append((None, members))
        rest = [e for e in rest if e not in members]
    return groups


def node(ax, x, y, color, closed, r=0.17):
    ax.add_patch(plt.Circle((x, y), r, facecolor=color, edgecolor=CLOSE if closed else "white",
                            linewidth=1.6 if closed else 0.8, zorder=3))


def draw(ax, groups, title, mark_closed, labels=False):
    x, cloud_y, fog_y, edge_y = 0.0, 2.55, 1.35, 0.25
    centers = []
    for fog, members in groups:
        w = max(len(members), 1) * 0.45 + 0.15
        ax.add_patch(FancyBboxPatch((x, edge_y - 0.3), w, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#f3f3f0", edgecolor=LINE, linewidth=0.8, zorder=1))
        for k, e in enumerate(members):
            node(ax, x + 0.3 + 0.45 * k, edge_y, EDGE, mark_closed and e in CLOSED)
        cx = x + w / 2
        if fog is not None:
            ax.plot([cx, cx], [fog_y - 0.17, edge_y + 0.3], color=LINE, lw=0.9, zorder=0)
            node(ax, cx, fog_y, FOG, mark_closed and fog in CLOSED, r=0.2)
        else:
            ax.text(cx, edge_y + 0.42, "no fog", ha="center", va="bottom", fontsize=5.5,
                    color=MUTED)
        centers.append((cx, fog))
        x += w + 0.2
    mid = (x - 0.2) / 2
    for cx, fog in centers:
        if fog is not None:
            ax.plot([cx, mid], [fog_y + 0.2, cloud_y - 0.25], color=LINE, lw=0.9, zorder=0)
    ax.add_patch(FancyBboxPatch((mid - 0.55, cloud_y - 0.25), 1.1, 0.5,
                                boxstyle="round,pad=0.02,rounding_size=0.12", facecolor="#f3f3f0",
                                edgecolor=LINE, linewidth=0.8, zorder=1))
    node(ax, mid - 0.22, cloud_y, CLOUD, False, r=0.17)
    node(ax, mid + 0.22, cloud_y, CLOUD, False, r=0.17)
    ax.text(mid + 0.65, cloud_y, "cloud", va="center", fontsize=6, color=MUTED)
    ax.text(-0.15, fog_y, "fog", va="center", ha="right", fontsize=6, color=MUTED)
    ax.text(-0.15, edge_y, "edge", va="center", ha="right", fontsize=6, color=MUTED)
    ax.text(-0.75, 3.45, title, fontsize=7, color=INK, va="top")
    if labels:
        cx0 = centers[0][0]
        ax.text(cx0 + 0.06, (fog_y + edge_y) / 2 + 0.05, "5 ms", fontsize=5.5, color=MUTED)
        ax.text(0.52, edge_y - 0.45, "2 ms", fontsize=5.5, color=MUTED, ha="center")
        ax.text((cx0 + mid) / 2 - 0.1, (fog_y + cloud_y) / 2 + 0.12, "30 ms", fontsize=5.5,
                color=MUTED, ha="right")
    ax.set_xlim(-0.75, 7.6)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    plt.rcParams.update({"font.family": "serif", "font.size": 7})
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.4, 3.5), dpi=300)
    draw(a, sites(SCEN / "continuum_tight.json"),
         "Baseline and reduced capacity (15 nodes)", True, labels=True)
    draw(b, sites(SCEN / "continuum_wide.json"), "Wider network (17 nodes)", False)
    handles = [plt.Line2D([], [], marker="o", ls="", markersize=5, markerfacecolor=c,
                          markeredgecolor=e, markeredgewidth=w)
               for c, e, w in ((CLOUD, "white", 0.5), (FOG, "white", 0.5), (EDGE, "white", 0.5),
                               ("white", CLOSE, 1.4))]
    fig.legend(handles, ["cloud node", "fog server (can sleep)", "edge device",
                         "closed 150–330 s (reduced capacity)"],
               loc="lower center", ncol=2, frameon=False, fontsize=6, handletextpad=0.3,
               columnspacing=1.0)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.12, hspace=0.25)
    fig.savefig(HERE / "topology.pdf")
    fig.savefig(HERE / "topology.png")


if __name__ == "__main__":
    main()
