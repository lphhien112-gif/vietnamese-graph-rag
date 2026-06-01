"""Smoke test sinh câu trả lời QUA proxy LLM (xirothedev) bằng đúng code path của project.

Chạy:  python -m tests.test_generation_live      (từ thư mục gốc repo)
hoặc:  pytest tests/test_generation_live.py -s -q

Mục đích: xác nhận .env (OPENAI_API_KEY + OPENAI_BASE_URL + VNGR_LLM_MODEL) load đúng và
Generator nối được tới LLM, sinh ra câu trả lời tiếng Việt grounding theo review giả lập.
"""

from __future__ import annotations

import sys
from pathlib import Path

# cho phép chạy trực tiếp không cần cài package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.rag.generate import Generator, build_prompt  # noqa: E402

# review giả lập (đứng thay cho kết quả retrieval, để test riêng phần sinh)
RETRIEVED = [
    {"source": "shopee", "text": "Camera chụp đêm của iPhone 15 rất nét, ít nhiễu, màu trung thực."},
    {"source": "uit-visfd", "text": "Pin dùng cả ngày thoải mái, sạc nhanh. Giá hơi cao so với cấu hình."},
    {"source": "shopee", "text": "Màn hình sáng đẹp, chơi game mượt, không nóng máy."},
]
GRAPH_CONTEXT = "iPhone 15 — CAMERA: 78% tích cực · BATTERY: 65% tích cực · PRICE: 40% tiêu cực (n=312 review)"
QUESTION = "Camera iPhone 15 chụp đêm có tốt không và pin dùng được lâu không?"


def run() -> int:
    cfg = Config.load()
    print(f"[cfg] model            = {cfg.llm.model}")
    print(f"[cfg] base_url (env)   = {__import__('os').environ.get('OPENAI_BASE_URL')}")
    print(f"[cfg] api_key present  = {bool(cfg.openai_api_key)}")

    gen = Generator(cfg)
    if not gen.available:
        print("\n❌ Generator KHÔNG khả dụng — kiểm tra OPENAI_API_KEY trong .env.")
        return 1

    prompt = build_prompt(QUESTION, RETRIEVED, GRAPH_CONTEXT)
    print("\n[prompt gửi LLM]\n" + "-" * 60)
    print(prompt)
    print("-" * 60)

    out = gen.generate(prompt)
    print("\n[câu trả lời LLM]\n" + "=" * 60)
    print(out["text"])
    print("=" * 60)
    print(f"\n[tokens] prompt={out['prompt_tokens']}  completion={out['completion_tokens']}")

    text = out["text"] or ""
    if text.startswith("(Lỗi LLM") or text.startswith("(LLM chưa cấu hình"):
        print("\n❌ FAIL — generation không thành công.")
        return 1
    print("\n✅ PASS — LLM sinh câu trả lời thành công qua proxy.")
    return 0


def test_generation_live():
    """Bản pytest: chỉ chạy khi có API key, ngược lại skip (an toàn cho CI no-secret)."""
    import os

    import pytest

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("Không có OPENAI_API_KEY — bỏ qua live test.")
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
