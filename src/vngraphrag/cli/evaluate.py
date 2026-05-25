"""Evaluation harness with a CI regression gate.

Computes Precision@k & MRR for an ablation of retrieval configs, using UIT-ViSFD gold
aspect labels as ground-truth (Shopee docs have no gold -> not counted). Writes
metrics.json and exits non-zero if the best config's MRR drops below cfg.eval_min_mrr.

    python -m vngraphrag.cli.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..core import DocumentIndex, aspect_from_query, aspects_from_text, maxsim
from ..rag.retrieval import _nz

EVAL_QUERIES = [
    ("camera chụp ảnh đẹp không", "CAMERA"),
    ("pin trâu dùng lâu không", "BATTERY"),
    ("màn hình hiển thị sắc nét", "SCREEN"),
    ("máy chạy mượt hiệu năng", "PERFORMANCE"),
    ("giá hợp lý hay đắt", "PRICE"),
    ("thiết kế đẹp cầm thoải mái", "DESIGN"),
    ("loa âm thanh tính năng", "FEATURES"),
    ("nhân viên tư vấn bảo hành dịch vụ", "SER&ACC"),
]
CONFIGS = {
    "bi_encoder": (1.0, 0.0, 0.0),
    "attention": (0.5, 0.5, 0.0),
    "graph_rag": (0.4, 0.4, 0.2),
}


def _components(query, index, encoder, n_cand=50):
    qv = encoder.encode_mean([query])[0]
    cand, sims = index.search(qv, n_cand)
    bi = sims[cand]
    q_tok = encoder.encode_tokens(query)
    attn = np.array([maxsim(q_tok, encoder.encode_tokens(index.records[i]["raw"])) for i in cand])
    q_asp = aspect_from_query(query)
    graph = np.array([1.0 if (q_asp and q_asp in aspects_from_text(index.records[i]["raw"])) else 0.0 for i in cand])
    return cand, bi, attn, graph


def _p_at_k(order, gold_sets, asp, k):
    return sum(1 for i in order[:k] if asp in gold_sets[i]) / k


def _mrr(order, gold_sets, asp):
    for r, i in enumerate(order, 1):
        if asp in gold_sets[i]:
            return 1.0 / r
    return 0.0


def run_eval(cfg: Config) -> dict:
    from ..core import PhoBERTEncoder

    encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("No index found. Run: python -m vngraphrag.cli.build_index")
    gold = [r["gold"] for r in index.records]

    cache = {q: _components(q, index, encoder) for q, _ in EVAL_QUERIES}
    results = {}
    for name, (wb, wa, wg) in CONFIGS.items():
        p5 = p10 = mr = 0.0
        for q, asp in EVAL_QUERIES:
            cand, bi, attn, graph = cache[q]
            comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph
            order = cand[comb.argsort()[::-1]]
            p5 += _p_at_k(order, gold, asp, 5)
            p10 += _p_at_k(order, gold, asp, 10)
            mr += _mrr(order, gold, asp)
        n = len(EVAL_QUERIES)
        results[name] = {"P@5": round(p5 / n, 4), "P@10": round(p10 / n, 4), "MRR": round(mr / n, 4)}
    return results


def run_clf_f1(cfg: Config) -> float | None:
    """Micro-F1 của BiLSTM aspect classifier trên tập dev (None nếu chưa deploy model)."""
    from ..core import load_visfd
    from ..core.aspect_clf import AspectClassifier

    clf = AspectClassifier.load(cfg.artifacts_dir)
    if clf is None:
        return None
    dev = load_visfd(cfg.data_dir, "Dev.csv")
    preds = clf.predict(list(dev["comment"]))
    tp = fp = fn = 0
    for gold, pred in zip(dev["aspects"], preds, strict=False):
        g, p = set(gold), set(pred)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0


def main():
    cfg = Config.load()
    results = run_eval(cfg)
    clf_f1 = run_clf_f1(cfg)
    out = Path(cfg.artifacts_dir) / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"retrieval": results, "aspect_clf_micro_f1": clf_f1}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    failed = False
    best_mrr = max(r["MRR"] for r in results.values())
    if best_mrr < cfg.eval_min_mrr:
        print(f"GATE FAIL: best MRR {best_mrr} < {cfg.eval_min_mrr}")
        failed = True
    else:
        print(f"Gate OK: best MRR {best_mrr} >= {cfg.eval_min_mrr}")
    if clf_f1 is not None:
        if clf_f1 < cfg.eval_min_f1:
            print(f"GATE FAIL: aspect-clf micro-F1 {clf_f1} < {cfg.eval_min_f1}")
            failed = True
        else:
            print(f"Gate OK: aspect-clf micro-F1 {clf_f1} >= {cfg.eval_min_f1}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
