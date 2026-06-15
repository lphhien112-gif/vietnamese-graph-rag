"""Evaluation harness với CI regression gate.

Ablation study 4 cấu hình truy xuất theo từng thành phần:
  bi_encoder → +attention (MaxSim) → +graph → +bm25 (full hybrid)

Metrics: Precision@5, Precision@10, MRR, NDCG@5 (4 chỉ số chuẩn IR).
Classifier: micro-F1 + macro-F1 + per-aspect F1 của BiLSTM.

Bộ 25 truy vấn đánh giá bao phủ đủ 10 khía cạnh UIT-ViSFD,
đảm bảo tính đại diện thống kê (vs. 8 câu cũ).

    python -m vngraphrag.cli.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..core import ASPECTS, DocumentIndex, aspect_from_query, aspects_from_text, maxsim
from ..rag.retrieval import _nz

# ── 100 eval queries — 10 câu/aspect (mở rộng từ 25 để siết khoảng tin cậy) ──
EVAL_QUERIES = [
    # CAMERA (10)
    ("camera chụp ảnh đẹp không", "CAMERA"),
    ("ảnh selfie có rõ không", "CAMERA"),
    ("chụp đêm có tốt không", "CAMERA"),
    ("zoom camera xa chụp có nét không", "CAMERA"),
    ("camera quay video có mượt không", "CAMERA"),
    ("ống kính chụp chân dung xoá phông đẹp không", "CAMERA"),
    ("camera trước chụp có sắc nét không", "CAMERA"),
    ("chụp ảnh thiếu sáng có bị nhiễu không", "CAMERA"),
    ("camera sau chụp cận cảnh thế nào", "CAMERA"),
    ("chất lượng ảnh chụp ban ngày ra sao", "CAMERA"),
    # BATTERY (10)
    ("pin trâu dùng lâu không", "BATTERY"),
    ("sạc pin nhanh không", "BATTERY"),
    ("pin có bền không tụt pin nhanh", "BATTERY"),
    ("dung lượng pin có lớn không", "BATTERY"),
    ("pin dùng được mấy tiếng một lần sạc", "BATTERY"),
    ("sạc pin có làm nóng máy không", "BATTERY"),
    ("pin có bị chai sau thời gian dùng không", "BATTERY"),
    ("pin xem phim chơi game tụt nhanh không", "BATTERY"),
    ("thời lượng pin dùng cả ngày có đủ không", "BATTERY"),
    ("pin sạc bao lâu thì đầy", "BATTERY"),
    # SCREEN (10)
    ("màn hình hiển thị sắc nét", "SCREEN"),
    ("cảm ứng có nhạy không", "SCREEN"),
    ("độ phân giải màn hình cao không", "SCREEN"),
    ("màn hình ngoài nắng có nhìn rõ không", "SCREEN"),
    ("tần số quét màn hình có mượt không", "SCREEN"),
    ("màu sắc màn hình hiển thị có đẹp không", "SCREEN"),
    ("màn hình có bị ám màu không", "SCREEN"),
    ("kích thước màn hình xem phim có đã không", "SCREEN"),
    ("độ sáng màn hình có đủ dùng không", "SCREEN"),
    ("viền màn hình có mỏng không", "SCREEN"),
    # PERFORMANCE (10)
    ("máy chạy mượt hiệu năng tốt", "PERFORMANCE"),
    ("chip mạnh không bị lag giật", "PERFORMANCE"),
    ("RAM xử lý nhanh mượt không", "PERFORMANCE"),
    ("chơi game nặng có giật lag không", "PERFORMANCE"),
    ("mở nhiều ứng dụng có mượt không", "PERFORMANCE"),
    ("cấu hình máy có mạnh không", "PERFORMANCE"),
    ("máy có bị đơ treo khi dùng không", "PERFORMANCE"),
    ("hiệu năng đa nhiệm có tốt không", "PERFORMANCE"),
    ("tốc độ xử lý có nhanh không", "PERFORMANCE"),
    ("chip xử lý đồ hoạ có khoẻ không", "PERFORMANCE"),
    # STORAGE (10)
    ("bộ nhớ trong bao nhiêu GB", "STORAGE"),
    ("dung lượng lưu trữ ROM có đủ không", "STORAGE"),
    ("bộ nhớ có lắp thẻ nhớ mở rộng được không", "STORAGE"),
    ("lưu được nhiều ảnh video không bộ nhớ", "STORAGE"),
    ("dung lượng bộ nhớ có lớn không", "STORAGE"),
    ("ROM còn trống nhiều không bộ nhớ", "STORAGE"),
    ("bộ nhớ trong có nhanh đầy không", "STORAGE"),
    ("máy có hỗ trợ thẻ nhớ ngoài không", "STORAGE"),
    ("dung lượng lưu trữ cài game có đủ không", "STORAGE"),
    ("bộ nhớ máy có đủ chứa dữ liệu không", "STORAGE"),
    # DESIGN (10)
    ("thiết kế đẹp cầm thoải mái", "DESIGN"),
    ("máy mỏng nhẹ kiểu dáng đẹp không", "DESIGN"),
    ("chất liệu vỏ máy có cao cấp không", "DESIGN"),
    ("màu sắc máy thiết kế có đẹp không", "DESIGN"),
    ("cầm máy có chắc tay không thiết kế", "DESIGN"),
    ("kiểu dáng có sang trọng không", "DESIGN"),
    ("mặt lưng máy thiết kế có đẹp không", "DESIGN"),
    ("trọng lượng máy mỏng nhẹ không", "DESIGN"),
    ("thiết kế có hợp thời trang không", "DESIGN"),
    ("máy nhìn ngoài thiết kế có đẹp không", "DESIGN"),
    # PRICE (10)
    ("giá hợp lý hay đắt so với chất lượng", "PRICE"),
    ("có đáng tiền không tầm giá này", "PRICE"),
    ("giá bán có rẻ không", "PRICE"),
    ("tầm giá này có đáng mua không", "PRICE"),
    ("mức giá có phù hợp túi tiền không", "PRICE"),
    ("giá thành so với cấu hình có hời không", "PRICE"),
    ("máy này giá có mắc quá không", "PRICE"),
    ("giá có cao hơn đắt hơn không", "PRICE"),
    ("bỏ tiền mua giá này có xứng đáng không", "PRICE"),
    ("giá khuyến mãi có rẻ tốt không", "PRICE"),
    # FEATURES (10)
    ("loa âm thanh chất lượng tính năng", "FEATURES"),
    ("wifi bluetooth NFC vân tay bảo mật", "FEATURES"),
    ("loa ngoài nghe nhạc có to không", "FEATURES"),
    ("cảm biến vân tay có nhạy không", "FEATURES"),
    ("có hỗ trợ NFC thanh toán không tính năng", "FEATURES"),
    ("bảo mật khuôn mặt có nhanh không", "FEATURES"),
    ("âm thanh loa xem phim có hay không", "FEATURES"),
    ("kết nối wifi sóng có ổn định không", "FEATURES"),
    ("có jack tai nghe cảm biến không", "FEATURES"),
    ("tính năng bảo mật vân tay có tốt không", "FEATURES"),
    # SER&ACC (10)
    ("nhân viên tư vấn bảo hành dịch vụ", "SER&ACC"),
    ("giao hàng đóng gói shop phục vụ", "SER&ACC"),
    ("bảo hành có nhanh không dịch vụ", "SER&ACC"),
    ("shop tư vấn có nhiệt tình không", "SER&ACC"),
    ("phụ kiện đi kèm có đầy đủ không", "SER&ACC"),
    ("dịch vụ hậu mãi bảo hành có tốt không", "SER&ACC"),
    ("giao hàng nhanh shop đóng gói cẩn thận", "SER&ACC"),
    ("đổi trả bảo hành có dễ không", "SER&ACC"),
    ("cửa hàng phục vụ có chu đáo không", "SER&ACC"),
    ("nhân viên giao hàng có thân thiện không", "SER&ACC"),
    # GENERAL (10)
    ("máy có tốt không nói chung tổng thể", "GENERAL"),
    ("sản phẩm ổn điện thoại dùng tốt", "GENERAL"),
    ("điện thoại này có đáng mua không", "GENERAL"),
    ("nhìn chung máy có ngon không", "GENERAL"),
    ("sản phẩm có chất lượng tốt không", "GENERAL"),
    ("tổng thể trải nghiệm điện thoại có tốt không", "GENERAL"),
    ("điện thoại dùng có ổn định không", "GENERAL"),
    ("có nên mua máy này không", "GENERAL"),
    ("máy này dùng tổng quan thế nào", "GENERAL"),
    ("sản phẩm điện thoại có đáng giới thiệu không", "GENERAL"),
]

# Ablation study: từng thành phần thêm vào
# (w_bi, w_attn, w_graph, w_bm25)
CONFIGS: dict[str, tuple[float, float, float, float]] = {
    "bi_encoder":   (1.0,  0.0,  0.0,  0.0),
    "+attention":   (0.5,  0.5,  0.0,  0.0),
    "+graph":       (0.4,  0.4,  0.2,  0.0),
    "+bm25 (full)": (0.35, 0.35, 0.15, 0.15),
}


def _components(query: str, index, encoder, bm25, n_cand: int = 50, tok_cache: dict | None = None):
    """Tính 4 score vectors cho 1 query trên top-n_cand dense candidates.

    `tok_cache` (dict doc_idx -> token-embedding) dùng chung giữa các truy vấn để
    KHÔNG encode lại token của cùng một doc nhiều lần (candidate lặp giữa các query)."""
    qv = encoder.encode_mean([query])[0]
    cand, sims = index.search(qv, n_cand)
    bi = sims[cand]

    def doc_tok(i: int):
        if tok_cache is None:
            return encoder.encode_tokens(index.records[i]["raw"])
        c = tok_cache.get(i)
        if c is None:
            c = encoder.encode_tokens(index.records[i]["raw"])
            tok_cache[i] = c
        return c

    q_tok = encoder.encode_tokens(query)
    attn = np.array([maxsim(q_tok, doc_tok(int(i))) for i in cand])

    q_asp = aspect_from_query(query)
    graph = np.array(
        [1.0 if (q_asp and q_asp in aspects_from_text(index.records[i]["raw"])) else 0.0
         for i in cand]
    )

    bm25_all = bm25.get_scores(query.lower().split())
    bm25_scores = bm25_all[cand]

    return cand, bi, attn, graph, bm25_scores


def _p_at_k(order: np.ndarray, gold_sets: list[set], asp: str, k: int) -> float:
    return sum(1 for i in order[:k] if asp in gold_sets[i]) / k


def _mrr(order: np.ndarray, gold_sets: list[set], asp: str) -> float:
    for r, i in enumerate(order, 1):
        if asp in gold_sets[i]:
            return 1.0 / r
    return 0.0


def _ndcg_at_k(order: np.ndarray, gold_sets: list[set], asp: str, k: int) -> float:
    """Binary-relevance NDCG@k."""
    dcg = sum(
        1.0 / np.log2(r + 2)
        for r, i in enumerate(order[:k])
        if asp in gold_sets[i]
    )
    n_rel = sum(1 for g in gold_sets if asp in g)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, n_rel)))
    return dcg / idcg if idcg > 0 else 0.0


def run_eval(cfg: Config) -> dict:
    """Chạy ablation retrieval, trả về dict metrics."""
    from rank_bm25 import BM25Okapi

    from ..core import PhoBERTEncoder

    encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("No index found. Run: python -m vngraphrag.cli.build_index")

    gold = [r["gold"] for r in index.records]

    # BM25 index một lần
    tokenized = [r["raw"].lower().split() for r in index.records]
    bm25 = BM25Okapi(tokenized)

    # Cache score components cho mỗi query (chia sẻ tok_cache giữa các query -> nhanh hơn)
    _tok_cache: dict = {}
    cache = {q: _components(q, index, encoder, bm25, tok_cache=_tok_cache) for q, _ in EVAL_QUERIES}

    results: dict = {}
    for name, (wb, wa, wg, wbm) in CONFIGS.items():
        p5 = p10 = mr = ndcg5 = 0.0
        per_aspect: dict[str, list[float]] = {a: [] for a in ASPECTS}

        for q, asp in EVAL_QUERIES:
            cand, bi, attn, graph, bm25_s = cache[q]
            comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph + wbm * _nz(bm25_s)
            order = cand[comb.argsort()[::-1]]

            p5 += _p_at_k(order, gold, asp, 5)
            p10 += _p_at_k(order, gold, asp, 10)
            mr += _mrr(order, gold, asp)
            ndcg5 += _ndcg_at_k(order, gold, asp, 5)

            if asp in per_aspect:
                per_aspect[asp].append(_mrr(order, gold, asp))

        n = len(EVAL_QUERIES)
        results[name] = {
            "P@5":    round(p5 / n, 4),
            "P@10":   round(p10 / n, 4),
            "MRR":    round(mr / n, 4),
            "NDCG@5": round(ndcg5 / n, 4),
            "per_aspect_mrr": {
                a: round(float(np.mean(v)), 4) if v else None
                for a, v in per_aspect.items()
            },
        }
    return results


def run_clf_f1(cfg: Config) -> dict | None:
    """Micro-F1 + Macro-F1 + per-aspect F1 của BiLSTM trên Dev set.
    Trả None nếu chưa có checkpoint."""
    from ..core import load_visfd
    from ..core.aspect_clf import AspectClassifier

    clf = AspectClassifier.load(cfg.artifacts_dir)
    if clf is None:
        return None

    dev = load_visfd(cfg.data_dir, "Dev.csv")
    preds = clf.predict(list(dev["comment"]))

    # Micro-F1
    tp_total = fp_total = fn_total = 0
    per_aspect: dict[str, dict] = {}

    for asp in ASPECTS:
        asp_tp = asp_fp = asp_fn = 0
        for gold_list, pred_set in zip(dev["aspects"], preds):
            g = asp in set(gold_list)
            p = asp in pred_set
            if g and p:
                asp_tp += 1
            elif not g and p:
                asp_fp += 1
            elif g and not p:
                asp_fn += 1
        tp_total += asp_tp
        fp_total += asp_fp
        fn_total += asp_fn

        prec = asp_tp / (asp_tp + asp_fp) if asp_tp + asp_fp else 0.0
        rec = asp_tp / (asp_tp + asp_fn) if asp_tp + asp_fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_aspect[asp] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }

    micro_prec = tp_total / (tp_total + fp_total) if tp_total + fp_total else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if tp_total + fn_total else 0.0
    micro_f1 = (
        round(2 * micro_prec * micro_rec / (micro_prec + micro_rec), 4)
        if micro_prec + micro_rec
        else 0.0
    )
    macro_f1 = round(float(np.mean([v["f1"] for v in per_aspect.values()])), 4)

    return {
        "micro_f1":   micro_f1,
        "macro_f1":   macro_f1,
        "per_aspect": per_aspect,
    }


def main():
    cfg = Config.load()
    print(f"Eval set: {len(EVAL_QUERIES)} queries × {len(CONFIGS)} configs")

    results = run_eval(cfg)
    clf_info = run_clf_f1(cfg)

    out = Path(cfg.artifacts_dir) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"retrieval": results, "aspect_clf": clf_info}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Bảng kết quả retrieval
    print(f"\n{'Config':18} {'P@5':>7} {'P@10':>7} {'MRR':>8} {'NDCG@5':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(
            f"{name:18} {r['P@5']:7.4f} {r['P@10']:7.4f}"
            f" {r['MRR']:8.4f} {r['NDCG@5']:8.4f}"
        )

    if clf_info:
        print(f"\nAspect classifier — micro-F1: {clf_info['micro_f1']:.4f}"
              f"  macro-F1: {clf_info['macro_f1']:.4f}")
        print(f"{'Aspect':12} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("-" * 36)
        for asp, v in clf_info["per_aspect"].items():
            print(f"{asp:12} {v['precision']:7.4f} {v['recall']:7.4f} {v['f1']:7.4f}")

    print(f"\n-> {out}")

    # CI regression gate
    failed = False
    best_mrr = max(r["MRR"] for r in results.values())
    if best_mrr < cfg.eval_min_mrr:
        print(f"GATE FAIL: best MRR {best_mrr:.4f} < {cfg.eval_min_mrr}")
        failed = True
    else:
        print(f"Gate OK: best MRR {best_mrr:.4f} >= {cfg.eval_min_mrr}")

    if clf_info is not None:
        micro = clf_info["micro_f1"]
        if micro < cfg.eval_min_f1:
            print(f"GATE FAIL: micro-F1 {micro:.4f} < {cfg.eval_min_f1}")
            failed = True
        else:
            print(f"Gate OK: micro-F1 {micro:.4f} >= {cfg.eval_min_f1}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
