"""Vẽ Knowledge Graph ra ảnh PNG (matplotlib, backend Agg — không cần màn hình).

Sinh 2 hình trong artifacts/:
  kg_overview.png        — toàn bộ đồ thị, tô màu theo loại node
  kg_brand_aspect.png    — tập trung Brand -> Aspect -> Sentiment (phần UIT-ViSFD)

Chạy:  python scripts/viz_kg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vngraphrag.core import load_kg  # noqa: E402

ART = ROOT / "artifacts"
G = load_kg(ART / "kg.pkl")

TYPE_COLOR = {
    "aspect": "#4C9BE0",
    "sentiment": "#9FD89F",
    "brand": "#F2C14E",
    "shop": "#C792EA",
    "product": "#E8C39E",
}


def legend(ax):
    ax.legend(
        handles=[mpatches.Patch(color=c, label=t) for t, c in TYPE_COLOR.items()],
        loc="upper right",
        fontsize=9,
    )


def color_of(n):
    return TYPE_COLOR.get(G.nodes[n].get("type"), "lightgray")


# ── Hình 1: tổng quan toàn đồ thị ─────────────────────────────────────────────
def overview():
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, k=0.55, seed=42, iterations=80)
    sizes = [350 + 90 * G.degree(n) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=[color_of(n) for n in G.nodes()], node_size=sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.18, ax=ax, arrows=False)
    # chỉ ghi nhãn node bậc cao (đỡ rối): aspect, brand, shop + product lớn
    labels = {
        n: (n if len(n) <= 22 else n[:20] + "…")
        for n, d in G.nodes(data=True)
        if d.get("type") in ("aspect", "brand", "shop") or G.degree(n) >= 4
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(f"Knowledge Graph — toàn cảnh ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", fontsize=15)
    legend(ax)
    ax.axis("off")
    out = ART / "kg_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ── Hình 2: Brand -> Aspect -> Sentiment (UIT-ViSFD) ──────────────────────────
def brand_aspect():
    keep = [n for n, d in G.nodes(data=True) if d.get("type") in ("brand", "aspect", "sentiment")]
    H = G.subgraph(keep).copy()
    fig, ax = plt.subplots(figsize=(16, 11))
    pos = nx.spring_layout(H, k=0.9, seed=7, iterations=120)
    nx.draw_networkx_nodes(H, pos, node_color=[color_of(n) for n in H.nodes()],
                           node_size=[500 + 70 * H.degree(n) for n in H.nodes()], ax=ax)
    # độ dày cạnh ~ log(weight)
    import math
    w = [0.4 + math.log1p(H[u][v]["weight"]) * 0.6 for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, width=w, alpha=0.3, ax=ax, arrows=True, arrowsize=8)
    nx.draw_networkx_labels(H, pos, font_size=8, ax=ax)
    ax.set_title("Brand → Aspect → Sentiment (UIT-ViSFD) — độ dày cạnh ~ số lần xuất hiện", fontsize=15)
    legend(ax)
    ax.axis("off")
    out = ART / "kg_brand_aspect.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    o1 = overview()
    o2 = brand_aspect()
    print(f"✅ Đã lưu:\n  {o1}\n  {o2}")
