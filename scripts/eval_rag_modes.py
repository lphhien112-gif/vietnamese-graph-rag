"""GAP 1 (lõi) — So sánh "KHÔNG dùng vs DÙNG" ở MỨC CÂU TRẢ LỜI.

Trả lời cùng một bộ câu hỏi bằng 3 chế độ, rồi chấm điểm:

  (A) No-RAG (closed-book): LLM trả lời TAY KHÔNG — không review, không Knowledge Graph.
      => đại diện "không dùng hệ thống". Đây là baseline để đo giá trị của retrieval.
  (B) Vanilla-RAG (dense thuần): chỉ PhoBERT bi-encoder top-k, KHÔNG graph, KHÔNG bm25/maxsim.
  (C) Graph-RAG (đầy đủ): hybrid retrieve + thống kê Knowledge Graph (hệ thống của dự án).

Chỉ số (mỗi câu trả lời):
  • faithfulness (0..100) — LLM-judge: tỉ lệ nội dung ĐƯỢC bằng chứng (review+KG) chứng minh.
    Bằng chứng tham chiếu = bằng chứng Graph-RAG (cùng một "sự thật corpus" cho cả 3 chế độ),
    nên No-RAG bị chấm thấp khi nói chung chung/bịa — chính là đo "giảm bịa đặt".
  • relevance (1..5) — LLM-judge: có thực sự trả lời đúng & hữu ích cho câu hỏi không.
  • numeric_grounding — chỉ số TỰ ĐỘNG (không qua LLM): % con số trong câu trả lời có trong
    bằng chứng. Bổ trợ cho LLM-judge (đỡ lệ thuộc một mình LLM chấm).

Câu hỏi NGOÀI PHẠM VI (adversarial): chỉ so No-RAG vs Graph-RAG ở khả năng TỪ CHỐI
(không bịa thông số IP68/Hz/GHz... mà review cảm xúc không có).

GIỚI HẠN trung thực: judge = gpt-4o-mini, cùng họ với generator -> có thể tự thiên vị.
Giảm thiểu bằng: chấm TỪNG câu độc lập (không so sánh cạnh nhau), bám chặt bằng chứng,
và đối chiếu với numeric_grounding tự động. Ghi rõ trong báo cáo.

Chạy:  python scripts/eval_rag_modes.py        (cần OPENAI_API_KEY)
Ghi:   artifacts/rag_modes_eval.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.rag.generate import SYSTEM, build_prompt  # noqa: E402
from vngraphrag.rag.pipeline import GraphRAGPipeline  # noqa: E402

# 30 câu hỏi GROUNDED — corpus có dữ liệu trả lời, phủ aspect + vài hãng cụ thể ──────
GROUNDED_Q = [
    "Camera điện thoại chụp đêm có tốt không?",
    "Chụp ảnh selfie có nét và sáng không?",
    "Camera iPhone chụp có đẹp không?",
    "Pin điện thoại dùng có lâu không?",
    "Pin Samsung có trâu không?",
    "Pin Xiaomi có bền không hay tụt nhanh?",
    "Sạc pin có nhanh không?",
    "Màn hình hiển thị có sắc nét không?",
    "Màn hình ngoài nắng có nhìn rõ không?",
    "Cảm ứng màn hình có nhạy không?",
    "Máy chạy có mượt không hay bị lag?",
    "Hiệu năng chơi game có ổn không?",
    "Chip xử lý có mạnh không?",
    "Bộ nhớ trong có đủ dùng không?",
    "Thiết kế máy có đẹp và cầm thoải mái không?",
    "Máy có mỏng nhẹ không?",
    "Giá bán có hợp lý so với chất lượng không?",
    "Tầm giá này mua có đáng tiền không?",
    "Loa ngoài nghe nhạc có to rõ không?",
    "Cảm biến vân tay mở khoá có nhanh không?",
    "Dịch vụ bảo hành có tốt không?",
    "Nhân viên tư vấn có nhiệt tình không?",
    "Shop giao hàng và đóng gói có cẩn thận không?",
    "Nhìn chung điện thoại có đáng mua không?",
    "Sản phẩm dùng tổng thể có ổn không?",
    "Camera Oppo chụp chân dung thế nào?",
    "Pin dùng cả ngày có hết không?",
    "Máy có nóng khi chơi game không?",
    "Chất lượng hoàn thiện máy có tốt không?",
    "Điện thoại này có hợp để quay video không?",
]

# 8 câu NGOÀI PHẠM VI — review cảm xúc KHÔNG chứa thông số -> hệ thống cần TỪ CHỐI ──
ADVERSARIAL_Q = [
    "iPhone 15 có chống nước đạt chuẩn IP68 không?",
    "Sạc đầy pin mất chính xác bao nhiêu phút?",
    "Màn hình có tần số quét đúng 144Hz không?",
    "Bảo hành chính hãng kéo dài mấy năm?",
    "Con chip bên trong xung nhịp bao nhiêu GHz?",
    "Máy nặng chính xác bao nhiêu gram?",
    "Pin có dung lượng chính xác bao nhiêu mAh?",
    "Camera có độ phân giải chính xác bao nhiêu chấm?",
]

ABSTAIN_CUES = [
    "không có thông tin", "không đề cập", "không nói", "chưa rõ", "không rõ",
    "không thể xác định", "dữ liệu không", "không tìm thấy", "không có dữ liệu",
    "không được nhắc", "không nêu", "không có đánh giá", "không đủ thông tin",
]
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> list[str]:
    out = []
    for m in NUM_RE.finditer(text):
        is_pct = m.end() < len(text) and text[m.end()] == "%"
        if is_pct or float(m.group().replace(",", ".")) >= 10:
            out.append(m.group())
    return out


def has_abstain(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in ABSTAIN_CUES)


def _gen(pipe: GraphRAGPipeline, prompt: str, tries: int = 4) -> str:
    """Gọi LLM, retry ngắn nếu lỗi tạm thời."""
    for k in range(tries):
        r = pipe.generator.generate(prompt)
        txt = r["text"]
        if not txt.startswith("(Lỗi LLM") and not txt.startswith("(LLM chưa"):
            return txt
        m = re.search(r"in (\d+)s", txt)
        wait = min(int(m.group(1)) if m else 8, 30)
        print(f"      retry {k + 1}/{tries} sau {wait}s ({txt[:50]})")
        time.sleep(wait)
    return txt


# ── 3 chế độ trả lời ──────────────────────────────────────────────────────────
def answer_norag(pipe: GraphRAGPipeline, q: str) -> str:
    prompt = (
        "Bạn là trợ lý tư vấn mua sắm online. Trả lời ngắn gọn bằng tiếng Việt.\n\n"
        f"=== CÂU HỎI ===\n{q}\n\nTrả lời:"
    )
    return _gen(pipe, prompt)


def retrieve_dense(pipe: GraphRAGPipeline, q: str, k: int = 5) -> list[dict]:
    """Vanilla RAG: chỉ dense bi-encoder top-k (không maxsim/bm25/graph)."""
    qv = pipe.encoder.encode_mean([q])[0]
    idx, _ = pipe.index.search(qv, k)
    out = []
    for i in idx:
        r = pipe.index.records[int(i)]
        out.append({"text": r["raw"], "source": r.get("source"), "product": r.get("product")})
    return out


def answer_vanilla(pipe: GraphRAGPipeline, q: str) -> tuple[str, str]:
    docs = retrieve_dense(pipe, q)
    prompt = build_prompt(q, docs, "")  # KHÔNG graph context
    evidence = " ".join(d["text"] for d in docs)
    return _gen(pipe, prompt), evidence


def answer_graph(pipe: GraphRAGPipeline, q: str) -> tuple[str, str]:
    res = pipe.answer(q)
    evidence = res["graph_context"] + " " + " ".join(r["text"] for r in res["retrieved"])
    return res["answer"], evidence


# ── LLM judge ─────────────────────────────────────────────────────────────────
JUDGE_SYS = "Bạn là giám khảo NGHIÊM KHẮC, công tâm, chấm điểm câu trả lời. Chỉ in JSON."


def judge(pipe: GraphRAGPipeline, question: str, answer: str, evidence: str) -> dict | None:
    prompt = (
        "Cho CÂU HỎI, BẰNG CHỨNG (trích từ review thật + thống kê), và CÂU TRẢ LỜI.\n"
        "Chấm 2 tiêu chí, CHỈ dựa trên BẰNG CHỨNG (đừng dùng kiến thức ngoài):\n"
        "  faithfulness (0-100): tỉ lệ nội dung câu trả lời ĐƯỢC bằng chứng chứng minh. "
        "Nếu câu trả lời nêu thông tin KHÔNG có trong bằng chứng (bịa/suy diễn) -> điểm THẤP. "
        "Câu trả lời chung chung không bám bằng chứng cũng thấp.\n"
        "  relevance (1-5): câu trả lời có trực tiếp & hữu ích cho câu hỏi không.\n\n"
        f"=== CÂU HỎI ===\n{question}\n\n=== BẰNG CHỨNG ===\n{evidence[:2000]}\n\n"
        f"=== CÂU TRẢ LỜI ===\n{answer}\n\n"
        'Chỉ in JSON: {"faithfulness": <0-100>, "relevance": <1-5>}'
    )
    for _ in range(3):
        try:
            resp = pipe.generator.client.chat.completions.create(
                model=pipe.generator.model,
                messages=[{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
            )
            txt = resp.choices[0].message.content
            m = re.search(r"\{[^{}]*\}", txt, re.S)
            if m:
                d = json.loads(m.group())
                return {"faithfulness": float(d["faithfulness"]), "relevance": float(d["relevance"])}
        except Exception as e:
            print(f"      judge err: {str(e)[:50]}")
            time.sleep(6)
    return None


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main() -> int:
    cfg = Config.load()
    print(f"[init] nạp pipeline... model = {cfg.llm.model}")
    pipe = GraphRAGPipeline.from_artifacts(cfg)
    if not pipe.generator.available:
        raise SystemExit("Không có OPENAI_API_KEY khả dụng.")

    records = []
    agg = {m: {"faith": [], "rel": [], "numok": 0, "numtot": 0}
           for m in ["no_rag", "vanilla_rag", "graph_rag"]}

    print(f"\n[GROUNDED] {len(GROUNDED_Q)} câu × 3 chế độ (+ judge):")
    for i, q in enumerate(GROUNDED_Q, 1):
        a_graph, ev = answer_graph(pipe, q)          # bằng chứng tham chiếu = của Graph-RAG
        a_van, _ = answer_vanilla(pipe, q)
        a_nor = answer_norag(pipe, q)
        row = {"type": "grounded", "question": q, "evidence": ev[:1500], "modes": {}}
        for mode, ans in [("no_rag", a_nor), ("vanilla_rag", a_van), ("graph_rag", a_graph)]:
            j = judge(pipe, q, ans, ev)
            nums = numbers(ans)
            ok = [n for n in nums if n in ev]
            agg[mode]["numtot"] += len(nums)
            agg[mode]["numok"] += len(ok)
            if j:
                agg[mode]["faith"].append(j["faithfulness"])
                agg[mode]["rel"].append(j["relevance"])
            row["modes"][mode] = {"answer": ans, "judge": j, "n_num": len(nums), "n_num_grounded": len(ok)}
        records.append(row)
        f = {m: (row["modes"][m]["judge"] or {}).get("faithfulness") for m in agg}
        print(f"  {i:2}/{len(GROUNDED_Q)} faith no/van/graph = "
              f"{f['no_rag']}/{f['vanilla_rag']}/{f['graph_rag']} | {q[:38]}")

    print(f"\n[ADVERSARIAL] {len(ADVERSARIAL_Q)} câu ngoài phạm vi — No-RAG vs Graph-RAG (cần TỪ CHỐI):")
    abst = {"no_rag": 0, "graph_rag": 0}
    for q in ADVERSARIAL_Q:
        a_graph, _ = answer_graph(pipe, q)
        a_nor = answer_norag(pipe, q)
        for mode, ans in [("no_rag", a_nor), ("graph_rag", a_graph)]:
            ab = has_abstain(ans)
            abst[mode] += int(ab)
            records.append({"type": "adversarial", "mode": mode, "question": q, "answer": ans, "abstained": ab})
        print(f"  từ chối no/graph = {has_abstain(a_nor)}/{has_abstain(a_graph)} | {q[:40]}")

    summary = {
        "n_grounded": len(GROUNDED_Q),
        "n_adversarial": len(ADVERSARIAL_Q),
        "model": cfg.llm.model,
        "judge_model": cfg.llm.model,
        "grounded": {
            m: {
                "faithfulness_mean": _avg(agg[m]["faith"]),
                "relevance_mean": _avg(agg[m]["rel"]),
                "numeric_grounding": (round(agg[m]["numok"] / agg[m]["numtot"], 4)
                                      if agg[m]["numtot"] else None),
                "n_numbers": agg[m]["numtot"],
            }
            for m in ["no_rag", "vanilla_rag", "graph_rag"]
        },
        "adversarial_abstention_rate": {
            "no_rag": round(abst["no_rag"] / len(ADVERSARIAL_Q), 4),
            "graph_rag": round(abst["graph_rag"] / len(ADVERSARIAL_Q), 4),
        },
        "note": "judge cùng họ generator (gpt-4o-mini) -> có thể tự thiên vị; đối chiếu numeric_grounding.",
    }
    out = Path(cfg.artifacts_dir) / "rag_modes_eval.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n=== TỔNG HỢP (GROUNDED) ===")
    print(f"{'Chế độ':14} {'Faithfulness':>13} {'Relevance':>10} {'NumGrounding':>13}")
    for m, lbl in [("no_rag", "No-RAG"), ("vanilla_rag", "Vanilla-RAG"), ("graph_rag", "Graph-RAG")]:
        g = summary["grounded"][m]
        print(f"{lbl:14} {str(g['faithfulness_mean']):>13} {str(g['relevance_mean']):>10} "
              f"{str(g['numeric_grounding']):>13}")
    print("\n=== ADVERSARIAL (tỉ lệ từ chối, cao = tốt) ===")
    print(f"  No-RAG   : {summary['adversarial_abstention_rate']['no_rag']}")
    print(f"  Graph-RAG: {summary['adversarial_abstention_rate']['graph_rag']}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
