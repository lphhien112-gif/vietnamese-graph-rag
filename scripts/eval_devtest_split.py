"""GAP 5 — Tách DEV/TEST cho retrieval + tune trọng số TRUNG THỰC (chống rò rỉ).

Vấn đề: tune_weights.py grid-search trọng số trên CHÍNH 25 truy vấn mà evaluate.py
dùng để báo cáo MRR -> "tuning trên tập test" -> con số bị thổi phồng.

Script này tách 25 truy vấn (phân tầng theo aspect, seed cố định) thành DEV/TEST:
  1. Grid-search trọng số trên DEV (tối đa MRR).
  2. Báo cáo 4 cấu hình ablation + cấu hình-tune-trên-DEV, tất cả CHẤM TRÊN TEST.
  3. Đối chiếu với "tune-trên-TEST" (con số rò rỉ) -> độ chênh = mức overfit.
Kèm khoảng tin cậy bootstrap 95% cho MRR trên TEST.

Chạy:  python scripts/eval_devtest_split.py
Ghi:   artifacts/devtest_eval.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import DocumentIndex  # noqa: E402
from vngraphrag.rag.retrieval import _nz  # noqa: E402
from vngraphrag.cli.evaluate import (  # noqa: E402
    CONFIGS,
    EVAL_QUERIES,
    _components,
    _mrr,
    _ndcg_at_k,
    _p_at_k,
)
from vngraphrag.cli.tune_weights import _weight_grid  # noqa: E402

RNG = np.random.default_rng(7)
N_BOOT = 2000


def stratified_split(queries):
    """Chia mỗi aspect ~nửa dev / nửa test (xen kẽ) -> cân bằng aspect, có thể tái lập."""
    by_asp: dict[str, list[int]] = {}
    for i, (_, asp) in enumerate(queries):
        by_asp.setdefault(asp, []).append(i)
    dev, test = [], []
    for asp, idxs in by_asp.items():
        order = list(RNG.permutation(idxs))
        for j, qi in enumerate(order):
            (dev if j % 2 == 0 else test).append(int(qi))
    return sorted(dev), sorted(test)


def mrr_per_query(weights, cache, gold, queries, idxs) -> list[float]:
    wb, wa, wg, wbm = weights
    out = []
    for qi in idxs:
        q, asp = queries[qi]
        cand, bi, attn, graph, bm25_s = cache[q]
        comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph + wbm * _nz(bm25_s)
        order = cand[comb.argsort()[::-1]]
        out.append(_mrr(order, gold, asp))
    return out


def full_metrics(weights, cache, gold, queries, idxs) -> dict:
    wb, wa, wg, wbm = weights
    p5 = p10 = mr = nd = 0.0
    for qi in idxs:
        q, asp = queries[qi]
        cand, bi, attn, graph, bm25_s = cache[q]
        comb = wb * _nz(bi) + wa * _nz(attn) + wg * graph + wbm * _nz(bm25_s)
        order = cand[comb.argsort()[::-1]]
        p5 += _p_at_k(order, gold, asp, 5)
        p10 += _p_at_k(order, gold, asp, 10)
        mr += _mrr(order, gold, asp)
        nd += _ndcg_at_k(order, gold, asp, 5)
    n = len(idxs)
    return {"P@5": round(p5 / n, 4), "P@10": round(p10 / n, 4),
            "MRR": round(mr / n, 4), "NDCG@5": round(nd / n, 4)}


def boot_ci(vals) -> list[float]:
    a = np.asarray(vals, dtype="float64")
    means = a[RNG.integers(0, len(a), size=(N_BOOT, len(a)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def best_on(idxs, grid, cache, gold, queries):
    scored = [(np.mean(mrr_per_query(w, cache, gold, queries, idxs)), w) for w in grid]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], float(scored[0][0])


def main() -> int:
    cfg = Config.load()
    from rank_bm25 import BM25Okapi

    from vngraphrag.core import PhoBERTEncoder

    encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("No index. Run: python -m vngraphrag.cli.build_index")
    gold = [r["gold"] for r in index.records]
    bm25 = BM25Okapi([r["raw"].lower().split() for r in index.records])

    print("Cache score components cho các query (PhoBERT, chia sẻ tok_cache)...")
    _tok_cache: dict = {}
    cache = {q: _components(q, index, encoder, bm25, tok_cache=_tok_cache) for q, _ in EVAL_QUERIES}

    dev, test = stratified_split(EVAL_QUERIES)
    print(f"DEV {len(dev)} câu | TEST {len(test)} câu\n")

    grid = _weight_grid()

    # (1) tune trên DEV -> chấm TEST (TRUNG THỰC)
    w_dev, dev_mrr = best_on(dev, grid, cache, gold, EVAL_QUERIES)
    # (2) tune trên TEST -> chấm TEST (RÒ RỈ, để đo overfit)
    w_test, test_tuned_mrr = best_on(test, grid, cache, gold, EVAL_QUERIES)

    out: dict = {"dev_idx": dev, "test_idx": test, "n_boot": N_BOOT, "configs_on_test": {}}

    # 4 cấu hình ablation cố định, chấm trên TEST
    for name, w in CONFIGS.items():
        met = full_metrics(w, cache, gold, EVAL_QUERIES, test)
        met["MRR_ci95"] = boot_ci(mrr_per_query(w, cache, gold, EVAL_QUERIES, test))
        out["configs_on_test"][name] = met

    # cấu hình tune-trên-DEV, chấm TEST
    met = full_metrics(w_dev, cache, gold, EVAL_QUERIES, test)
    met["MRR_ci95"] = boot_ci(mrr_per_query(w_dev, cache, gold, EVAL_QUERIES, test))
    met["weights"] = list(w_dev)
    out["configs_on_test"]["tuned_on_DEV (honest)"] = met

    out["overfit_check"] = {
        "tuned_on_DEV_weights": list(w_dev),
        "MRR_dev_at_dev_weights": round(dev_mrr, 4),
        "MRR_test_at_dev_weights": out["configs_on_test"]["tuned_on_DEV (honest)"]["MRR"],
        "tuned_on_TEST_weights": list(w_test),
        "MRR_test_at_test_weights (leaky)": round(test_tuned_mrr, 4),
        "overfit_gap_MRR": round(test_tuned_mrr - out["configs_on_test"]["tuned_on_DEV (honest)"]["MRR"], 4),
    }

    op = Path(cfg.artifacts_dir) / "devtest_eval.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'Config (chấm trên TEST)':26} {'P@5':>6} {'P@10':>6} {'MRR':>7} {'NDCG@5':>7} {'MRR 95% CI':>16}")
    print("-" * 78)
    for name, m in out["configs_on_test"].items():
        ci = m["MRR_ci95"]
        print(f"{name:26} {m['P@5']:6.3f} {m['P@10']:6.3f} {m['MRR']:7.3f} {m['NDCG@5']:7.3f}"
              f"   [{ci[0]:.3f},{ci[1]:.3f}]")
    oc = out["overfit_check"]
    print(f"\nOverfit check: tune-trên-TEST MRR={oc['MRR_test_at_test_weights (leaky)']} (rò rỉ) "
          f"vs tune-trên-DEV→TEST MRR={oc['MRR_test_at_dev_weights']} (trung thực) "
          f"-> gap={oc['overfit_gap_MRR']}")
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
