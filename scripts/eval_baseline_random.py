"""Baseline NGẪU NHIÊN (sàn tham chiếu) cho bài toán retrieval — để thấy Graph RAG
(MRR 0.854) cao hơn mức 'đoán mò' bao nhiêu. Xếp hạng ngẫu nhiên toàn corpus, lặp R lần
lấy trung bình. Dùng cùng 8 truy vấn + cùng định nghĩa relevance (nhãn vàng aspect).

Chạy:  python scripts/eval_baseline_random.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.cli.evaluate import EVAL_QUERIES, _mrr, _p_at_k  # noqa: E402
from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import DocumentIndex  # noqa: E402

R = 200  # số lần lặp lấy trung bình


def main() -> int:
    cfg = Config.load()
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("Chưa có index.")
    gold = [r["gold"] for r in index.records]
    n = len(gold)
    rng = np.random.RandomState(42)

    p5 = p10 = mr = 0.0
    for _ in range(R):
        for _, asp in EVAL_QUERIES:
            order = rng.permutation(n)
            p5 += _p_at_k(order, gold, asp, 5)
            p10 += _p_at_k(order, gold, asp, 10)
            mr += _mrr(order, gold, asp)
    denom = R * len(EVAL_QUERIES)
    res = {"P@5": round(p5 / denom, 4), "P@10": round(p10 / denom, 4), "MRR": round(mr / denom, 4)}
    out = Path(cfg.artifacts_dir) / "baseline_random.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Baseline ngẫu nhiên ({R} lần lặp, {n} docs): {res}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
