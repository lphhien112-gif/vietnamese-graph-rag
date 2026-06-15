"""Grid search để tìm bộ trọng số tối ưu cho Hybrid Retriever.

Tìm (w_bi, w_attn, w_graph, w_bm25) tối đa hóa MRR trên 25-query eval set.
Nếu --update-config: ghi kết quả tốt nhất vào config.yaml.

    python -m vngraphrag.cli.tune_weights
    python -m vngraphrag.cli.tune_weights --update-config
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np

from ..config import Config
from ..core import DocumentIndex, aspect_from_query, aspects_from_text, maxsim
from ..rag.retrieval import _nz
from .evaluate import EVAL_QUERIES, _mrr, _ndcg_at_k, _p_at_k, _components

# Grid: mỗi chiều nhận một trong các giá trị này
_GRID_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]

# Chỉ giữ tổ hợp có tổng = 1.0 (±0.01)
def _weight_grid() -> list[tuple[float, float, float, float]]:
    combos = []
    for wb, wa, wg, wbm in product(_GRID_VALUES, repeat=4):
        if abs(wb + wa + wg + wbm - 1.0) < 0.015:
            combos.append((wb, wa, wg, wbm))
    return combos


def _score_config(
    weights: tuple[float, float, float, float],
    cache: dict,
    gold: list[set],
    metric: str = "MRR",
) -> float:
    wb, wa, wg, wbm = weights
    total = 0.0
    for q, asp in EVAL_QUERIES:
        cand, bi, attn, graph, bm25_s = cache[q]
        comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph + wbm * _nz(bm25_s)
        order = cand[comb.argsort()[::-1]]
        if metric == "MRR":
            total += _mrr(order, gold, asp)
        elif metric == "NDCG@5":
            total += _ndcg_at_k(order, gold, asp, 5)
        else:
            total += _p_at_k(order, gold, asp, 5)
    return total / len(EVAL_QUERIES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--metric", default="MRR", choices=["MRR", "NDCG@5", "P@5"],
        help="Metric tối ưu hóa"
    )
    ap.add_argument("--top-k", type=int, default=10, help="Hiển thị top-K kết quả")
    ap.add_argument(
        "--update-config", action="store_true",
        help="Ghi bộ trọng số tốt nhất vào config.yaml"
    )
    args = ap.parse_args()

    from rank_bm25 import BM25Okapi

    from ..core import PhoBERTEncoder

    cfg = Config.load(args.config)
    encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("No index. Run: python -m vngraphrag.cli.build_index")

    gold = [r["gold"] for r in index.records]
    tokenized = [r["raw"].lower().split() for r in index.records]
    bm25 = BM25Okapi(tokenized)

    print("Cache score components cho 25 queries...")
    cache = {q: _components(q, index, encoder, bm25) for q, _ in EVAL_QUERIES}

    grid = _weight_grid()
    print(f"Grid search: {len(grid)} tổ hợp trọng số — tối ưu {args.metric}\n")

    scored = []
    for w in grid:
        s = _score_config(w, cache, gold, args.metric)
        scored.append((s, w))
    scored.sort(reverse=True)

    print(f"{'Rank':>5} {'w_bi':>6} {'w_attn':>7} {'w_graph':>8} {'w_bm25':>7} {args.metric:>8}")
    print("-" * 50)
    for rank, (s, (wb, wa, wg, wbm)) in enumerate(scored[: args.top_k], 1):
        print(f"{rank:5} {wb:6.2f} {wa:7.2f} {wg:8.2f} {wbm:7.2f} {s:8.4f}")

    best_score, best_w = scored[0]
    print(f"\nBest: w_bi={best_w[0]}  w_attn={best_w[1]}  "
          f"w_graph={best_w[2]}  w_bm25={best_w[3]}  {args.metric}={best_score:.4f}")

    out = Path(cfg.artifacts_dir) / "tune_weights.json"
    out.write_text(
        json.dumps(
            {
                "metric": args.metric,
                "best": {
                    "w_bi": best_w[0], "w_attn": best_w[1],
                    "w_graph": best_w[2], "w_bm25": best_w[3],
                    args.metric: best_score,
                },
                "top10": [
                    {"w_bi": w[0], "w_attn": w[1], "w_graph": w[2], "w_bm25": w[3], args.metric: s}
                    for s, w in scored[:10]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"-> {out}")

    if args.update_config:
        import yaml  # type: ignore[import-untyped]

        p = Path(args.config)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        raw.setdefault("retrieval", {}).update(
            w_bi=best_w[0], w_attn=best_w[1], w_graph=best_w[2], w_bm25=best_w[3]
        )
        p.write_text(yaml.dump(raw, allow_unicode=True, default_flow_style=False), encoding="utf-8")
        print(f"Updated config: {p}")


if __name__ == "__main__":
    main()
