"""Demo FULL Graph RAG pipeline: retrieve (PhoBERT bi-encoder + MaxSim + graph boost)
-> graph context (KG) -> sinh câu trả lời (LLM qua proxy).

Chạy:  python scripts/demo_rag.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.rag.pipeline import GraphRAGPipeline  # noqa: E402

QUESTIONS = [
    "Camera điện thoại chụp đêm có tốt không?",
    "Pin Samsung dùng có lâu không?",
    "Dịch vụ bảo hành và nhân viên tư vấn thế nào?",
]


def main() -> int:
    cfg = Config.load()
    print(f"[init] model={cfg.llm.model} · nạp index + KG từ {cfg.artifacts_dir}/ ...")
    pipe = GraphRAGPipeline.from_artifacts(cfg)
    print(f"[init] index v{pipe.index.version} · {len(pipe.index.records)} docs · KG {pipe.kg.number_of_nodes()} nodes\n")

    for q in QUESTIONS:
        print("=" * 70)
        print(f"❓ {q}")
        res = pipe.answer(q)
        print("\n📚 Top review truy xuất (retrieve):")
        for r in res["retrieved"][:3]:
            print(f"   [{r.get('source')}] score={r.get('score', 0):.3f}  {r['text'][:90]}")
        print("\n📊 Graph context (KG):")
        print("   " + (res["graph_context"].strip().replace("\n", "\n   ") or "(không có)"))
        print("\n💬 Câu trả lời (LLM):")
        print("   " + (res["answer"] or "").replace("\n", "\n   "))
        print(f"\n⏱️  latency={res['latency_ms']}ms · cost=${res['cost_usd']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
