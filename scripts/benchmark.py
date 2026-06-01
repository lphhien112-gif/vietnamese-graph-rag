"""Benchmark harness — đo & THEO DÕI hiệu năng hệ thống theo thời gian.

Khác với evaluate.py (chỉ regression gate cho CI), file này tạo một BÁO CÁO BENCHMARK
đầy đủ + so sánh với baseline đã lưu để phát hiện cải thiện/thoái hoá:

  Chất lượng : P@5/P@10/MRR cho 3 cấu hình retrieval (ablation) + micro-F1 BiLSTM
  Tốc độ     : latency trung vị/p95 của encode query, retrieve, end-to-end (n lần lặp)
  Quy mô     : #docs index, #node/#edge KG, kích thước artifacts

  python scripts/benchmark.py                 # chạy + in báo cáo + so với baseline
  python scripts/benchmark.py --save-baseline # lưu kết quả hiện tại làm baseline mới
  python scripts/benchmark.py --runs 10       # số lần lặp đo latency (mặc định 5)

Kết quả -> artifacts/benchmark.json ; baseline -> artifacts/benchmark_baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402

# truy vấn đo latency (đa dạng aspect)
LATENCY_QUERIES = [
    "camera chụp đêm có tốt không",
    "pin dùng được bao lâu",
    "màn hình sắc nét không",
    "giá có hợp lý không",
    "dịch vụ bảo hành thế nào",
]


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def measure_quality(cfg) -> dict:
    """P@k/MRR ablation + micro-F1 — tái dùng evaluate.py (cùng định nghĩa, không lệch)."""
    from vngraphrag.cli.evaluate import run_clf_f1, run_eval

    retrieval = run_eval(cfg)
    f1 = run_clf_f1(cfg)
    best_mrr = max(r["MRR"] for r in retrieval.values())
    return {"retrieval": retrieval, "best_mrr": best_mrr, "aspect_clf_micro_f1": f1}


def measure_speed(cfg, runs: int) -> dict:
    """Latency encode query / retrieve / end-to-end (ms)."""
    from vngraphrag.core import PhoBERTEncoder
    from vngraphrag.rag.pipeline import GraphRAGPipeline

    pipe = GraphRAGPipeline.from_artifacts(cfg)
    enc: PhoBERTEncoder = pipe.encoder

    enc_ms, retr_ms = [], []
    # warmup (nạp lazy, JIT cache)
    enc.encode_mean([LATENCY_QUERIES[0]])
    pipe.retriever.retrieve(LATENCY_QUERIES[0])

    for _ in range(runs):
        for q in LATENCY_QUERIES:
            t0 = time.perf_counter()
            enc.encode_mean([q])
            enc_ms.append((time.perf_counter() - t0) * 1000)
            t1 = time.perf_counter()
            pipe.retriever.retrieve(q)
            retr_ms.append((time.perf_counter() - t1) * 1000)

    return {
        "n_measurements": len(enc_ms),
        "encode_query_ms": {"median": round(statistics.median(enc_ms), 2), "p95": _pct(enc_ms, 0.95)},
        "retrieve_ms": {"median": round(statistics.median(retr_ms), 2), "p95": _pct(retr_ms, 0.95)},
    }


def measure_scale(cfg) -> dict:
    from vngraphrag.core import DocumentIndex, load_kg

    art = Path(cfg.artifacts_dir)
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    out = {"n_docs": len(index.records) if index else 0}
    kg_path = art / "kg.pkl"
    if kg_path.exists():
        G = load_kg(kg_path)
        out["kg_nodes"] = G.number_of_nodes()
        out["kg_edges"] = G.number_of_edges()
    sizes = {}
    for f in ["doc_vectors.npy", "kg.pkl", "aspect_clf.pt", "records.json"]:
        p = art / f
        if p.exists():
            sizes[f] = f"{p.stat().st_size // 1024} KB"
    out["artifact_sizes"] = sizes
    return out


def compare_baseline(current: dict, baseline: dict) -> list[str]:
    """So sánh chất lượng với baseline; cảnh báo nếu tụt > 1% tương đối."""
    lines = []
    cb, bb = current["quality"], baseline.get("quality", {})
    for key in ["best_mrr", "aspect_clf_micro_f1"]:
        cv, bv = cb.get(key), bb.get(key)
        if cv is None or bv is None:
            continue
        delta = cv - bv
        arrow = "→" if abs(delta) < 1e-6 else ("↑" if delta > 0 else "↓")
        flag = "  ⚠️ REGRESSION" if delta < -0.01 * bv else ""
        lines.append(f"  {key:22} {bv:.4f} {arrow} {cv:.4f}  (Δ{delta:+.4f}){flag}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--no-speed", action="store_true", help="bỏ qua đo latency (nhanh, chỉ chất lượng)")
    args = ap.parse_args()

    cfg = Config.load()
    art = Path(cfg.artifacts_dir)
    if not (art / "manifest.json").exists():
        raise SystemExit("Chưa có index. Chạy: python -m vngraphrag.cli.build_index")

    print("⏳ Đo chất lượng (P@k/MRR/F1)...")
    quality = measure_quality(cfg)
    print("⏳ Đo quy mô...")
    scale = measure_scale(cfg)
    speed = {}
    if not args.no_speed:
        print(f"⏳ Đo tốc độ ({args.runs} lần lặp × {len(LATENCY_QUERIES)} truy vấn)...")
        speed = measure_speed(cfg, args.runs)

    result = {"quality": quality, "speed": speed, "scale": scale, "model": cfg.embedding_model,
              "llm_model": cfg.llm.model}

    (art / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- in báo cáo ----
    print("\n" + "=" * 64)
    print("📊 BENCHMARK REPORT")
    print("=" * 64)
    print(f"\n▸ CHẤT LƯỢNG  (best MRR = {quality['best_mrr']:.4f}, BiLSTM F1 = {quality['aspect_clf_micro_f1']})")
    for name, r in quality["retrieval"].items():
        print(f"    {name:12} P@5={r['P@5']:.3f}  P@10={r['P@10']:.3f}  MRR={r['MRR']:.3f}")
    if speed:
        print(f"\n▸ TỐC ĐỘ  (median / p95, {speed['n_measurements']} phép đo)")
        print(f"    encode query : {speed['encode_query_ms']['median']:7.1f} / {speed['encode_query_ms']['p95']:7.1f} ms")
        print(f"    retrieve     : {speed['retrieve_ms']['median']:7.1f} / {speed['retrieve_ms']['p95']:7.1f} ms")
    print("\n▸ QUY MÔ")
    print(f"    docs={scale['n_docs']}  KG={scale.get('kg_nodes','?')} node/{scale.get('kg_edges','?')} cạnh")
    print(f"    artifacts: {scale['artifact_sizes']}")

    # ---- so baseline ----
    bpath = art / "benchmark_baseline.json"
    if bpath.exists() and not args.save_baseline:
        baseline = json.loads(bpath.read_text(encoding="utf-8"))
        print("\n▸ SO VỚI BASELINE")
        for line in compare_baseline(result, baseline):
            print(line)
    elif not bpath.exists():
        print("\n(chưa có baseline — chạy với --save-baseline để tạo)")

    if args.save_baseline:
        bpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ Đã lưu baseline -> {bpath}")

    print(f"\n-> {art / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
