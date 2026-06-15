"""LLM generation via OpenAI. Builds the grounded prompt and returns answer + token usage.
Degrades gracefully when no API key is configured."""

from __future__ import annotations

SYSTEM = "Bạn là trợ lý tư vấn mua sắm online, trả lời bằng tiếng Việt."


def build_prompt(question: str, retrieved: list[dict], graph_context: str) -> str:
    ctx = "\n".join(f"[{r.get('source')}] {r['text'][:240]}" for r in retrieved)
    return (
        "Bạn là trợ lý tư vấn mua sắm online. CHỈ dựa trên dữ liệu dưới đây, "
        "trả lời ngắn gọn bằng tiếng Việt.\n"
        "QUAN TRỌNG: nếu dữ liệu KHÔNG chứa thông số kỹ thuật được hỏi "
        "(vd. IP68, Hz, GHz, mAh, gram, số năm bảo hành, phút sạc, độ phân giải), "
        "PHẢI trả lời 'Đánh giá không đề cập thông tin này' — TUYỆT ĐỐI không bịa "
        "thông số từ kiến thức ngoài.\n\n"
        f"=== ĐÁNH GIÁ NGƯỜI DÙNG ===\n{ctx}\n\n"
        f"=== THỐNG KÊ TỪ KNOWLEDGE GRAPH ===\n{graph_context or 'Không có thống kê.'}\n\n"
        f"=== CÂU HỎI ===\n{question}\n\nTrả lời:"
    )


class Generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.client = None
        self.model = cfg.llm.model
        key = cfg.openai_api_key
        if key:
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=key)
            except Exception:
                self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def generate(self, prompt: str) -> dict:
        if self.client is None:
            return {"text": "(LLM chưa cấu hình — xem review trích dẫn)", "prompt_tokens": 0, "completion_tokens": 0}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=self.cfg.llm.temperature,
                max_tokens=self.cfg.llm.max_tokens,
            )
            u = resp.usage
            return {
                "text": resp.choices[0].message.content,
                "prompt_tokens": getattr(u, "prompt_tokens", 0),
                "completion_tokens": getattr(u, "completion_tokens", 0),
            }
        except Exception as e:
            return {"text": f"(Lỗi LLM: {e})", "prompt_tokens": 0, "completion_tokens": 0}
