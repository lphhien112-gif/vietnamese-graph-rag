"""Unit test cho build_prompt (no-API): grounding + chỉ thị abstention (M6 fix).

build_prompt là hàm thuần, không gọi OpenAI (Generator mới import openai ở __init__),
nên chạy được trong CI light không cần API key.
"""

from vngraphrag.rag.generate import build_prompt


def test_build_prompt_has_abstention_instruction():
    """M6 fix: prompt ép từ chối khi dữ liệu không có thông số kỹ thuật được hỏi."""
    p = build_prompt(
        "Máy có chống nước chuẩn IP68 không?",
        [{"source": "UIT-ViSFD", "text": "pin dùng tốt, máy đẹp"}],
        "",
    )
    assert "không đề cập" in p.lower()  # câu từ chối bắt buộc
    assert "IP68" in p  # ví dụ thông số cứng được liệt kê
    assert "CHỈ dựa trên dữ liệu" in p  # vẫn giữ grounding gốc


def test_build_prompt_includes_evidence_and_question():
    p = build_prompt(
        "pin thế nào",
        [{"source": "UIT-ViSFD", "text": "pin trâu dùng cả ngày"}],
        "BATTERY: 56% tích cực",
    )
    assert "pin trâu" in p and "pin thế nào" in p and "56% tích cực" in p
