"""Sinh hình cho báo cáo TỪ SỐ LIỆU THẬT (khớp các bảng trong report):

  report/figures/embedding_comparison.png  <- artifacts/embeddings_full.json  (2 panel + CI)
  report/figures/rag_modes.png             <- artifacts/rag_modes_eval.json   (No-RAG/RAG/Graph)
  report/figures/ablation.png              <- artifacts/metrics.json          (ablation retrieval)
  report/figures/aspect_f1.png             <- artifacts/metrics.json          (F1 từng aspect)
  report/figures/knowledge_graph.png       <- copy từ artifacts/kg_brand_aspect.png

Chạy sau khi đã chạy các script eval. Hình nào thiếu dữ liệu sẽ được bỏ qua (in cảnh báo).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _load(name):
    p = ART / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


done = []

# ── embedding_comparison.png (2 panel: keyword-rich vs lexical-gap, có CI) ─────
d = _load("embeddings_full.json")
if d:
    methods = ["Keyword", "TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]
    colors = ["#bbbbbb", "#6BAED6", "#74C476", "#9E9AC8", "#FD8D3C"]
    titles = {"keyword_rich": "Truy vấn GIÀU từ khoá (25 câu)",
              "lexical_gap": "Truy vấn LEXICAL-GAP (16 câu)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, (sname, title) in zip(axes, titles.items(), strict=False):
        sr = d["sets"][sname]
        mrr = [sr[m]["MRR"] for m in methods]
        cis = [sr[m]["MRR_ci95"] for m in methods]
        lo = [m - c[0] for m, c in zip(mrr, cis, strict=False)]
        hi = [c[1] - m for m, c in zip(mrr, cis, strict=False)]
        ax.bar(methods, mrr, color=colors, yerr=[lo, hi], capsize=4)
        for i, v in enumerate(mrr):
            ax.text(i, min(v + 0.03, 1.0), f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("MRR (± 95% CI bootstrap)")
    fig.suptitle("Keyword baseline vs học biểu diễn — KHÔNG dùng vs DÙNG ngữ nghĩa (pool 10.923)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "embedding_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    done.append("embedding_comparison.png")
else:
    print("⚠ thiếu embeddings_full.json -> bỏ embedding_comparison.png")

# ── rag_modes.png (No-RAG vs Vanilla-RAG vs Graph-RAG) ────────────────────────
rm = _load("rag_modes_eval.json")
if rm and "summary" in rm:
    g = rm["summary"]["grounded"]
    modes = [("no_rag", "No-RAG"), ("vanilla_rag", "Vanilla-RAG"), ("graph_rag", "Graph-RAG")]
    faith = [g[m]["faithfulness_mean"] or 0 for m, _ in modes]
    ng = [(g[m]["numeric_grounding"] or 0) * 100 for m, _ in modes]
    rel = [(g[m]["relevance_mean"] or 0) * 20 for m, _ in modes]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(modes))
    w = 0.26
    ax.bar(x - w, faith, w, label="Faithfulness (0-100)", color="#FD8D3C")
    ax.bar(x, rel, w, label="Relevance (×20)", color="#74C476")
    ax.bar(x + w, ng, w, label="Numeric grounding (%)", color="#6BAED6")
    for i, v in enumerate(faith):
        ax.text(x[i] - w, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in modes])
    ax.set_ylim(0, 108)
    ax.set_title("KHÔNG dùng vs DÙNG (mức câu trả lời): faithfulness tăng dần theo retrieval + graph")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "rag_modes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    done.append("rag_modes.png")
else:
    print("⚠ thiếu rag_modes_eval.json -> bỏ rag_modes.png")

# ── ablation.png + aspect_f1.png ──────────────────────────────────────────────
m = _load("metrics.json")
if m and "retrieval" in m:
    r = m["retrieval"]
    names = list(r.keys())
    rng = _load("baseline_random.json") or {}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(names))
    w = 0.38
    mrr = [r[n]["MRR"] for n in names]
    ndcg = [r[n].get("NDCG@5", 0) for n in names]
    ax.bar(x - w / 2, mrr, w, label="MRR", color="#FD8D3C")
    ax.bar(x + w / 2, ndcg, w, label="NDCG@5", color="#6BAED6")
    for i, v in enumerate(mrr):
        ax.text(x[i] - w / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
    if rng.get("MRR"):
        ax.axhline(rng["MRR"], ls="--", color="gray", lw=1)
        ax.text(len(names) - 0.5, rng["MRR"] + 0.006, f"sàn ngẫu nhiên {rng['MRR']:.3f}",
                ha="right", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Ablation Retrieval (100 truy vấn)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ablation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    done.append("ablation.png")

    pa = (m.get("aspect_clf") or {}).get("per_aspect")
    if pa:
        asps = list(pa.keys())
        f1 = [pa[a]["f1"] for a in asps]
        cols = ["#d62728" if v < 0.5 else "#74C476" for v in f1]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.bar(asps, f1, color=cols)
        for i, v in enumerate(f1):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        macro = m["aspect_clf"].get("macro_f1")
        ax.axhline(macro, ls="--", color="gray", lw=1)
        ax.set_title(f"BiLSTM F1 từng aspect (đỏ = F1<0.5; gạch = macro-F1 {macro})")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "aspect_f1.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        done.append("aspect_f1.png")

# ── knowledge_graph.png (copy) ────────────────────────────────────────────────
src = ART / "kg_brand_aspect.png"
if src.exists():
    shutil.copy(src, FIG / "knowledge_graph.png")
    done.append("knowledge_graph.png")

print("✅ Đã sinh:", ", ".join(done) if done else "(không có hình nào)")
