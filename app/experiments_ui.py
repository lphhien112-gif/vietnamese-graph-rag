"""Experiment Dashboard — giao diện trình bày & chạy thí nghiệm Vietnamese Graph RAG.

Khác với app/ui.py (chỉ hỏi-đáp 1 câu), dashboard này phục vụ phần RESEARCH:

  Tab 1 — 📊 Kết quả thí nghiệm: nạp các JSON trong artifacts/ và vẽ bảng + biểu đồ
          (ablation retrieval, embedding + baseline Keyword với khoảng tin cậy,
           No-RAG vs Vanilla-RAG vs Graph-RAG, F1 từng aspect, grounding).
  Tab 2 — 🔬 So sánh trực tiếp: nhập 1 câu hỏi → chạy SONG SONG 3 chế độ
          (No-RAG / Vanilla-RAG / Graph-RAG) để thấy tận mắt giá trị của retrieval + graph.
  Tab 3 — 🕸️ Demo hỏi-đáp: pipeline đầy đủ + Knowledge Graph context + review trích dẫn.

    python -m app.experiments_ui
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import gradio as gr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

_PIPE = None


def _load(name: str):
    p = ART / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _empty_fig(msg: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="#888")
    ax.axis("off")
    return fig


# ── Tab 1 figures ─────────────────────────────────────────────────────────────
def fig_ablation():
    m = _load("metrics.json")
    if not m or "retrieval" not in m:
        return _empty_fig("Chưa có metrics.json — chạy: python -m vngraphrag.cli.evaluate")
    r = m["retrieval"]
    names = list(r.keys())
    rng = _load("baseline_random.json") or {}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(names))
    w = 0.38
    mrr = [r[n]["MRR"] for n in names]
    ndcg = [r[n].get("NDCG@5", 0) for n in names]
    b1 = ax.bar(x - w / 2, mrr, w, label="MRR", color="#FD8D3C")
    b2 = ax.bar(x + w / 2, ndcg, w, label="NDCG@5", color="#6BAED6")
    for bars, vals in [(b1, mrr), (b2, ndcg)]:
        for b, v in zip(bars, vals, strict=False):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    if rng.get("MRR"):
        ax.axhline(rng["MRR"], ls="--", color="gray", lw=1)
        ax.text(len(names) - 0.5, rng["MRR"] + 0.005, f"sàn ngẫu nhiên {rng['MRR']:.3f}",
                ha="right", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Ablation Retrieval (100 truy vấn): đóng góp tăng dần từng thành phần")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_embeddings():
    d = _load("embeddings_full.json")
    if not d:
        return _empty_fig("Chưa có embeddings_full.json — chạy: python scripts/eval_embeddings_full.py")
    methods = ["Keyword", "TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]
    sets = d.get("sets", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    titles = {"keyword_rich": "Truy vấn GIÀU từ khoá", "lexical_gap": "Truy vấn LEXICAL-GAP (không trùng từ)"}
    for ax, (sname, title) in zip(axes, titles.items(), strict=False):
        sr = sets.get(sname, {})
        if not sr:
            ax.text(0.5, 0.5, "chưa có", ha="center")
            ax.axis("off")
            continue
        mrr = [sr.get(me, {}).get("MRR", 0) for me in methods]
        cis = [sr.get(me, {}).get("MRR_ci95", [0, 0]) for me in methods]
        lo = [m - c[0] for m, c in zip(mrr, cis, strict=False)]
        hi = [c[1] - m for m, c in zip(mrr, cis, strict=False)]
        colors = ["#bbbbbb", "#6BAED6", "#74C476", "#9E9AC8", "#FD8D3C"]
        ax.bar(methods, mrr, color=colors, yerr=[lo, hi], capsize=4)
        for i, v in enumerate(mrr):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].set_ylabel("MRR (± 95% CI bootstrap)")
    fig.suptitle("Keyword baseline vs học biểu diễn — KHÔNG dùng vs DÙNG ngữ nghĩa", fontsize=11)
    fig.tight_layout()
    return fig


def fig_rag_modes():
    d = _load("rag_modes_eval.json")
    if not d or "summary" not in d:
        return _empty_fig("Chưa có rag_modes_eval.json — chạy: python scripts/eval_rag_modes.py")
    g = d["summary"]["grounded"]
    modes = [("no_rag", "No-RAG"), ("vanilla_rag", "Vanilla-RAG"), ("graph_rag", "Graph-RAG")]
    faith = [g[m]["faithfulness_mean"] or 0 for m, _ in modes]
    rel = [(g[m]["relevance_mean"] or 0) * 20 for m, _ in modes]  # 1-5 -> %ish scale
    ng = [(g[m]["numeric_grounding"] or 0) * 100 for m, _ in modes]
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
    ax.set_ylim(0, 105)
    ax.set_title("KHÔNG dùng vs DÙNG (mức câu trả lời): No-RAG → Vanilla-RAG → Graph-RAG")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_classifier():
    m = _load("metrics.json")
    if not m or not m.get("aspect_clf", {}).get("per_aspect"):
        return _empty_fig("Chưa có per-aspect F1 — chạy: python -m vngraphrag.cli.evaluate")
    pa = m["aspect_clf"]["per_aspect"]
    asps = list(pa.keys())
    f1 = [pa[a]["f1"] for a in asps]
    colors = ["#d62728" if v < 0.5 else "#74C476" for v in f1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(asps, f1, color=colors)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    micro = m["aspect_clf"].get("micro_f1")
    macro = m["aspect_clf"].get("macro_f1")
    ax.axhline(macro, ls="--", color="gray", lw=1)
    ax.set_title(f"BiLSTM F1 từng aspect — micro={micro}  macro={macro} (đỏ = F1<0.5, điểm yếu micro che giấu)")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()
    return fig


def table_summary() -> pd.DataFrame:
    rows = []
    m = _load("metrics.json")
    if m:
        r = m["retrieval"]
        best = max(r, key=lambda k: r[k]["MRR"])
        rows.append(["Retrieval — cấu hình tốt nhất", best, f"MRR {r[best]['MRR']}  NDCG@5 {r[best].get('NDCG@5')}"])
        rows.append(["Retrieval — bi-encoder thuần", "bi_encoder", f"MRR {r['bi_encoder']['MRR']}"])
        ac = m.get("aspect_clf") or {}
        rows.append(["BiLSTM aspect", "micro/macro-F1", f"{ac.get('micro_f1')} / {ac.get('macro_f1')}"])
    rm = _load("rag_modes_eval.json")
    if rm:
        g = rm["summary"]["grounded"]
        rows.append(["Faithfulness No-RAG → Graph-RAG", "0-100",
                     f"{g['no_rag']['faithfulness_mean']} → {g['graph_rag']['faithfulness_mean']}"])
        ab = rm["summary"]["adversarial_abstention_rate"]
        rows.append(["Từ chối câu ngoài phạm vi", "No-RAG / Graph-RAG", f"{ab['no_rag']} / {ab['graph_rag']}"])
    gr_ = _load("grounding_eval.json")
    if gr_ and gr_.get("summary", {}).get("numeric_grounding_rate") is not None:
        s = gr_["summary"]
        rows.append(["Numeric-grounding (TN grounding)", "tỉ lệ", str(s["numeric_grounding_rate"])])
    dv = _load("devtest_eval.json")
    if dv:
        oc = dv["overfit_check"]
        rows.append(["Overfit do tuning (MRR test)", "rò rỉ vs trung thực",
                     f"{oc['MRR_test_at_test_weights (leaky)']} vs {oc['MRR_test_at_dev_weights']} (gap {oc['overfit_gap_MRR']})"])
    if not rows:
        rows = [["Chưa có kết quả", "—", "chạy các script trong scripts/ + cli.evaluate"]]
    return pd.DataFrame(rows, columns=["Hạng mục", "Cấu hình", "Kết quả"])


def refresh_tab1():
    return table_summary(), fig_ablation(), fig_embeddings(), fig_rag_modes(), fig_classifier()


# ── Tab 2: live 3-mode comparison ─────────────────────────────────────────────
def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from vngraphrag.config import Config
        from vngraphrag.rag.pipeline import GraphRAGPipeline

        _PIPE = GraphRAGPipeline.from_artifacts(Config.load())
    return _PIPE


def compare_modes(question: str):
    if not question.strip():
        return "Nhập câu hỏi.", "", "", ""
    from eval_rag_modes import answer_graph, answer_norag, answer_vanilla  # type: ignore

    sys.path.insert(0, str(ROOT / "scripts"))
    pipe = _get_pipe()
    a_graph, ev = answer_graph(pipe, question)
    a_van, _ = answer_vanilla(pipe, question)
    a_nor = answer_norag(pipe, question)
    note = ("**No-RAG** trả lời tay không (dễ chung chung/bịa). "
            "**Vanilla-RAG** bám review nhưng thiếu thống kê. "
            "**Graph-RAG** thêm thống kê Knowledge Graph (bằng chứng dưới).")
    return a_nor, a_van, a_graph, f"{note}\n\n**Bằng chứng (Graph-RAG):**\n\n{ev[:1200]}"


# ── Tab 3: full demo ──────────────────────────────────────────────────────────
def demo_answer(question: str):
    if not question.strip():
        return "", "", ""
    r = _get_pipe().answer(question)
    refs = "\n\n".join(
        f"[{d.get('source')}{(' · ' + str(d['product'])) if d.get('product') else ''}] {d['text'][:160]}"
        for d in r.get("retrieved", [])
    )
    meta = f"⏱️ {r.get('latency_ms', 0)} ms · 💵 ${r.get('cost_usd', 0)} · id `{r.get('id','')}`"
    return r.get("answer", ""), r.get("graph_context", "") + "\n\n" + meta, refs


def build():
    with gr.Blocks(title="Vietnamese Graph RAG — Experiment Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔬 Vietnamese Graph RAG — Experiment Dashboard\n"
                    "Trình bày & chạy thí nghiệm: ablation retrieval, baseline Keyword, "
                    "No-RAG vs RAG vs Graph-RAG, grounding, dev/test.")

        with gr.Tab("📊 Kết quả thí nghiệm"):
            gr.Markdown("Số liệu nạp trực tiếp từ `artifacts/*.json`. Bấm **Tải lại** sau khi chạy script.")
            btn = gr.Button("🔄 Tải lại kết quả", variant="primary")
            tbl = gr.Dataframe(value=table_summary(), label="Bảng tổng hợp", wrap=True)
            with gr.Row():
                p_abl = gr.Plot(fig_ablation(), label="Ablation retrieval")
                p_emb = gr.Plot(fig_embeddings(), label="Embedding + Keyword baseline")
            with gr.Row():
                p_rag = gr.Plot(fig_rag_modes(), label="No-RAG vs RAG vs Graph-RAG")
                p_clf = gr.Plot(fig_classifier(), label="F1 từng aspect")
            btn.click(refresh_tab1, outputs=[tbl, p_abl, p_emb, p_rag, p_clf])

        with gr.Tab("🔬 So sánh trực tiếp (live)"):
            gr.Markdown("Nhập câu hỏi → chạy **3 chế độ** để thấy tận mắt: KHÔNG dùng vs DÙNG retrieval/graph.")
            q2 = gr.Textbox(label="Câu hỏi", value="Pin Samsung dùng có lâu không?")
            b2 = gr.Button("So sánh 3 chế độ", variant="primary")
            with gr.Row():
                o_nor = gr.Textbox(label="① No-RAG (tay không)")
                o_van = gr.Textbox(label="② Vanilla-RAG (dense)")
                o_gra = gr.Textbox(label="③ Graph-RAG (đầy đủ)")
            ev2 = gr.Markdown()
            b2.click(compare_modes, inputs=q2, outputs=[o_nor, o_van, o_gra, ev2])

        with gr.Tab("🕸️ Demo hỏi-đáp"):
            q3 = gr.Textbox(label="Câu hỏi", value="Camera chụp đêm có tốt không?")
            b3 = gr.Button("Hỏi", variant="primary")
            a3 = gr.Textbox(label="💬 Trả lời")
            with gr.Row():
                kg3 = gr.Textbox(label="🕸️ Knowledge Graph context + meta")
                rf3 = gr.Textbox(label="📄 Review trích dẫn")
            gr.Examples([["Pin dùng được bao lâu?"], ["Máy nào giá rẻ mà camera tốt?"],
                         ["Shop giao hàng có nhanh không?"]], inputs=q3)
            b3.click(demo_answer, inputs=q3, outputs=[a3, kg3, rf3])

    return demo


if __name__ == "__main__":
    build().launch(share=True)
