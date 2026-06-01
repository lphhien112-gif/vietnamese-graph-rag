"""Test FastAPI serving layer in-process bằng TestClient (kích hoạt lifespan -> nạp pipeline).
Gọi đủ 4 endpoint: /health /classify /query /feedback.

Chạy:  python scripts/test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # để import package `app` ở gốc repo

from app.api import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def main() -> int:
    print("[api] khởi tạo TestClient (lifespan nạp index + KG + LLM)...")
    with TestClient(app) as c:
        h = c.get("/health").json()
        print("GET  /health   ->", h)
        assert h["status"] == "ok"

        cl = c.post("/classify", json={"text": "Pin tụt nhanh, camera chụp đêm mờ, giá lại đắt"}).json()
        print("POST /classify ->", cl)
        assert "aspects" in cl

        q = c.post("/query", json={"question": "Màn hình có sắc nét không?"}).json()
        print("POST /query    -> answer:", (q.get("answer") or "")[:120])
        print("                  latency_ms:", q.get("latency_ms"), "| n_retrieved:", len(q.get("retrieved", [])))
        qid = q.get("id")
        assert qid

        fb = c.post("/feedback", json={"query_id": qid, "rating": 1, "note": "demo"}).json()
        print("POST /feedback ->", fb)
        assert fb.get("ok")

    print("\n✅ PASS — cả 4 endpoint FastAPI hoạt động.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
