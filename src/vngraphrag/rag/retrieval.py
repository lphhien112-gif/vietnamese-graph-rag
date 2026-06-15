"""Hybrid retrieval: PhoBERT bi-encoder ∪ BM25 sparse → union candidates
→ MaxSim (ColBERT-style) rerank + Knowledge Graph aspect boost.

Kiến trúc 4 thành phần:
  (1) Dense  — PhoBERT cosine similarity, top-N candidates
  (2) Sparse — BM25 Okapi, top-N candidates (union với dense)
  (3) MaxSim — token-level late interaction (ColBERT-style), chỉ tính cho dense candidates
  (4) Graph  — aspect keyword boost từ Knowledge Graph (không dùng nhãn vàng → không rò rỉ)

Union giúp bắt exact-match (mã sản phẩm, số model) mà dense bỏ sót;
MaxSim giúp rerank ngữ nghĩa; Graph boost tín hiệu cấu trúc.
"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from ..core import aspect_from_query, aspects_from_text, maxsim


def _nz(x: np.ndarray) -> np.ndarray:
    """Min-max normalize về [0, 1]; trả 0 nếu range quá nhỏ."""
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else x * 0.0


class HybridRetriever:
    def __init__(self, index, encoder, cfg):
        self.index = index
        self.encoder = encoder
        self.cfg = cfg
        # Xây BM25 index một lần từ toàn bộ corpus
        tokenized = [r["raw"].lower().split() for r in index.records]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        weights: tuple[float, float, float, float] | None = None,
    ) -> list[dict]:
        c = self.cfg.retrieval
        top_k = top_k or c.top_k
        w_bi, w_attn, w_graph, w_bm25 = weights or (c.w_bi, c.w_attn, c.w_graph, c.w_bm25)

        # ── (1) Dense bi-encoder: top-N candidates + toàn bộ similarity scores ──
        q_mean = self.encoder.encode_mean([query])[0]
        dense_cand, all_sims = self.index.search(q_mean, c.n_candidates)

        # ── (2) BM25 sparse: top-N candidates ──
        bm25_all = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_all)[::-1][: c.n_candidates]

        # ── Union: lên đến 2×N candidates ──
        cand = np.union1d(dense_cand, bm25_top)

        # Score bi-encoder và BM25 cho toàn bộ union (cả hai là full arrays)
        bi = all_sims[cand]
        bm25_cand = bm25_all[cand]

        # ── (3) MaxSim chỉ trên dense candidates (token encoding tốn kém) ──
        q_tok = self.encoder.encode_tokens(query)
        dense_set = set(dense_cand.tolist())
        attn_map: dict[int, float] = {}
        for i in dense_cand:
            attn_map[int(i)] = maxsim(
                q_tok, self.encoder.encode_tokens(self.index.records[int(i)]["raw"])
            )
        attn = np.array([attn_map.get(int(i), 0.0) for i in cand])

        # ── (4) Graph boost: keyword aspect matching ──
        q_asp = aspect_from_query(query)
        graph = np.array(
            [
                1.0
                if (q_asp and q_asp in aspects_from_text(self.index.records[int(i)]["raw"]))
                else 0.0
                for i in cand
            ]
        )

        # ── Kết hợp có trọng số (min-max normalize từng thành phần) ──
        combined = (
            w_bi * _nz(bi)
            + w_attn * _nz(attn)
            + w_graph * graph
            + w_bm25 * _nz(bm25_cand)
        )
        order = combined.argsort()[::-1][:top_k]

        results = []
        for o in order:
            i = int(cand[o])
            r = self.index.records[i]
            results.append(
                {
                    "idx": i,
                    "text": r["raw"],
                    "source": r.get("source"),
                    "product": r.get("product"),
                    "shop": r.get("shop"),
                    "rating": r.get("rating"),
                    "score": float(combined[o]),
                }
            )
        return results
