"""GAP 3 — Đo GROUNDING (chống bịa đặt) bằng số, thay cho '3 demo chấm mắt'.

Hai chỉ số (đều tự động, không cần nhãn người):

1) numeric-grounding rate: mọi CON SỐ (đặc biệt %/đếm review) trong câu trả lời phải
   xuất hiện trong BẰNG CHỨNG đưa cho LLM (graph_context + review trích xuất). Con số
   bịa (không có trong bằng chứng) = hallucination. Đây là loại tuyên bố dễ kiểm chứng
   nhất nên dùng làm thước đo chính.

2) abstention rate trên câu hỏi NGOÀI PHẠM VI: các câu hỏi đòi thông số cụ thể mà review
   cảm xúc KHÔNG chứa (IP68, số năm bảo hành, phút sạc, Hz...). Hệ thống grounded phải
   NÓI 'không có thông tin' thay vì bịa thông số.

GIỚI HẠN (khai báo trung thực): numeric-grounding là PROXY — nó bắt số bịa, KHÔNG bắt
được mọi kiểu bịa diễn đạt (paraphrase). Vẫn tốt hơn nhiều so với 3 demo chấm tay.

Chạy:  python scripts/eval_grounding.py        (cần OPENAI_API_KEY + proxy)
Ghi:   artifacts/grounding_eval.json
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
from vngraphrag.rag.pipeline import GraphRAGPipeline  # noqa: E402

# Câu hỏi GROUNDED-được (corpus có dữ liệu trả lời) ──────────────────────────────
GROUNDED_Q = [
    "Camera điện thoại chụp đêm có tốt không?",
    "Pin Samsung dùng có lâu không?",
    "Pin Xiaomi có bền không?",
    "Màn hình hiển thị có đẹp không?",
    "Máy chạy có mượt không?",
    "Giá cả thế nào, có hợp lý không?",
    "Dịch vụ bảo hành và nhân viên tư vấn ra sao?",
    "Thiết kế máy có đẹp không?",
]

# Câu hỏi NGOÀI PHẠM VI — đòi thông số cụ thể review cảm xúc KHÔNG có ──────────────
# (test: hệ thống có từ chối/không bịa thông số không)
ADVERSARIAL_Q = [
    "iPhone 15 có chống nước đạt chuẩn IP68 không?",
    "Sạc đầy pin mất chính xác bao nhiêu phút?",
    "Màn hình có tần số quét đúng 144Hz không?",
    "Bảo hành chính hãng kéo dài mấy năm?",
    "Con chip bên trong xung nhịp bao nhiêu GHz?",
    "Máy nặng chính xác bao nhiêu gram?",
]

ABSTAIN_CUES = [
    "không có thông tin", "không đề cập", "không nói", "chưa rõ", "không rõ",
    "không thể xác định", "dữ liệu không", "không tìm thấy", "không có dữ liệu",
    "không được nhắc", "không nêu", "không có đánh giá", "không đủ thông tin",
]

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def numbers(text: str) -> list[str]:
    """Các con số 'đáng kiểm' trong câu trả lời: phần trăm hoặc số nguyên >= 10
    (bỏ qua '1 ngày', '2 năm'... vốn hay là trải nghiệm thường, ít rủi ro bịa thống kê)."""
    out = []
    for m in NUM_RE.finditer(text):
        tok = m.group()
        is_pct = m.end() < len(text) and text[m.end()] == "%"
        val = float(tok.replace(",", "."))
        if is_pct or val >= 10:
            out.append(tok)
    return out


def grounded_in(num: str, evidence: str) -> bool:
    return num in evidence


def has_abstain(text: str) -> bool:
    low = text.lower()
    return any(c in low for c in ABSTAIN_CUES)


def answer_with_retry(pipe: GraphRAGPipeline, q: str, max_wait_s: int = 650) -> dict:
    """Gọi pipeline; nếu proxy trả rate-limit (503) thì backoff theo gợi ý rồi thử lại."""
    waited = 0
    while True:
        res = pipe.answer(q)
        txt = res["answer"]
        if not txt.startswith("(Lỗi LLM") and not txt.startswith("(LLM chưa"):
            return res
        # đọc gợi ý 'try again in Ns' nếu có
        m = re.search(r"in (\d+)s", txt)
        wait = min(int(m.group(1)) if m else 60, 320)
        if waited + wait > max_wait_s:
            res["answer"] = "(LLM_UNAVAILABLE sau backoff)"
            return res
        print(f"    rate-limited, chờ {wait}s rồi thử lại... ({q[:30]}...)")
        time.sleep(wait)
        waited += wait


def selftest_metrics() -> None:
    """Kiểm chứng LOGIC đo bằng dữ liệu giả (không cần LLM) — chứng minh harness đúng."""
    # answer có 1 số grounded (56) và 1 số bịa (99) so với bằng chứng
    ev = "BATTERY/Positive: 126 review (54%)  pin dùng 2 ngày rất tốt"
    nums = numbers("Pin tốt, 54% tích cực nhưng có người nói 99% chê.")
    assert set(nums) == {"54", "99"}, nums
    assert grounded_in("54", ev) and not grounded_in("99", ev)
    assert has_abstain("Xin lỗi, dữ liệu không đề cập thông tin này.")
    assert not has_abstain("Có, máy chống nước chuẩn IP68.")
    print("✓ selftest_metrics: logic numeric-grounding + abstention hoạt động đúng.\n")


def preflight(pipe: GraphRAGPipeline) -> bool:
    """1 lệnh gọi thử. False nếu LLM/proxy không phục vụ (rate-limit/no_accounts)."""
    r = pipe.generator.generate("Trả lời 1 từ: ok?")
    txt = r["text"]
    if txt.startswith("(Lỗi LLM") or txt.startswith("(LLM chưa"):
        print(f"[preflight] LLM KHÔNG khả dụng -> {txt[:120]}")
        return False
    return True


def main() -> int:
    cfg = Config.load()
    selftest_metrics()
    print(f"[init] nạp pipeline (PhoBERT + index + KG)... model LLM = {cfg.llm.model}")
    pipe = GraphRAGPipeline.from_artifacts(cfg)

    if not preflight(pipe):
        status = {"status": "BLOCKED_LLM_UNAVAILABLE",
                  "reason": "proxy trả 503/no_accounts — không có tài khoản phục vụ",
                  "harness": "đã sẵn sàng; chạy lại scripts/eval_grounding.py khi API hồi phục",
                  "model": cfg.llm.model}
        out = Path(cfg.artifacts_dir) / "grounding_eval.json"
        out.write_text(json.dumps({"summary": status, "records": []}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n⚠ Dừng: LLM proxy không khả dụng. Ghi trạng thái -> {out}")
        print("  (Harness đã verify đúng qua selftest; chỉ thiếu API để ra số thật.)")
        return 0

    records = []
    n_num = n_num_ok = 0

    print(f"\n[GROUNDED] {len(GROUNDED_Q)} câu hỏi (corpus trả lời được):")
    for q in GROUNDED_Q:
        res = answer_with_retry(pipe, q)
        ans = res["answer"]
        evidence = res["graph_context"] + " " + " ".join(r["text"] for r in res["retrieved"])
        nums = numbers(ans)
        ok = [n for n in nums if grounded_in(n, evidence)]
        n_num += len(nums)
        n_num_ok += len(ok)
        records.append({
            "type": "grounded", "question": q, "answer": ans,
            "numbers": nums, "numbers_grounded": ok,
            "all_numbers_grounded": len(nums) == len(ok),
        })
        flag = "✓" if len(nums) == len(ok) else "✗ BỊA SỐ"
        print(f"  {flag}  số={len(ok)}/{len(nums)} grounded | {q[:42]}")

    print(f"\n[ADVERSARIAL] {len(ADVERSARIAL_Q)} câu hỏi ngoài phạm vi (cần TỪ CHỐI):")
    n_abstain = 0
    for q in ADVERSARIAL_Q:
        res = answer_with_retry(pipe, q)
        ans = res["answer"]
        abst = has_abstain(ans)
        n_abstain += int(abst)
        records.append({
            "type": "adversarial", "question": q, "answer": ans, "abstained": abst,
        })
        print(f"  {'✓ từ chối' if abst else '✗ KHÔNG từ chối'} | {q[:42]} -> {ans[:60]}")

    summary = {
        "numeric_grounding_rate": round(n_num_ok / n_num, 4) if n_num else None,
        "n_numbers_total": n_num,
        "n_numbers_grounded": n_num_ok,
        "abstention_rate_adversarial": round(n_abstain / len(ADVERSARIAL_Q), 4),
        "n_grounded_q": len(GROUNDED_Q),
        "n_adversarial_q": len(ADVERSARIAL_Q),
        "model": cfg.llm.model,
        "note": "numeric-grounding là proxy (bắt số bịa, không bắt mọi paraphrase-hallucination)",
    }
    out = Path(cfg.artifacts_dir) / "grounding_eval.json"
    out.write_text(json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== TỔNG HỢP GROUNDING ===")
    print(f"  numeric-grounding rate : {summary['numeric_grounding_rate']}  "
          f"({n_num_ok}/{n_num} con số có trong bằng chứng)")
    print(f"  abstention (ngoài phạm vi): {summary['abstention_rate_adversarial']}  "
          f"({n_abstain}/{len(ADVERSARIAL_Q)} câu từ chối đúng)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
