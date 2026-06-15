"""Phân tích độ ĐỦ của dữ liệu Knowledge Graph: liệu KG có đủ dày để trả lời
truy vấn cảm xúc theo hãng/aspect không, hay quá thưa ở đâu.

Ghi: artifacts/kg_stats.json  + in bảng.
Chạy: python scripts/analyze_kg.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import ASPECTS, load_kg  # noqa: E402

MIN_OK = 30   # ngưỡng "đủ" cho một thống kê cảm xúc theo hãng×aspect


def main() -> int:
    cfg = Config.load()
    G = load_kg(Path(cfg.artifacts_dir) / "kg.pkl")

    by_type: dict[str, int] = defaultdict(int)
    for _, d in G.nodes(data=True):
        by_type[d.get("type", "?")] += 1

    brands = [n for n, d in G.nodes(data=True) if d.get("type") == "brand"]
    products = [n for n, d in G.nodes(data=True) if d.get("type") == "product"]

    # corpus-wide aspect totals (aspect -> sentiment weight)
    aspect_total: dict[str, int] = {}
    for a in ASPECTS:
        if a in G:
            aspect_total[a] = sum(
                G[a][nb]["weight"] for nb in G.successors(a)
                if G.nodes[nb].get("type") == "sentiment"
            )

    # brand -> aspect#sentiment: tổng theo hãng & theo (hãng, aspect)
    brand_total: dict[str, int] = {}
    brand_aspect: dict[str, dict[str, int]] = {}
    for b in brands:
        per_asp: dict[str, int] = defaultdict(int)
        for nb in G.successors(b):
            d = G.nodes[nb]
            if d.get("type") == "sentiment" and "#" in nb:
                asp = nb.split("#", 1)[0]
                per_asp[asp] += G[b][nb]["weight"]
        brand_aspect[b] = dict(per_asp)
        brand_total[b] = sum(per_asp.values())

    brand_sorted = sorted(brand_total.items(), key=lambda x: -x[1])
    n_brand_ge100 = sum(1 for _, v in brand_sorted if v >= 100)
    n_brand_ge30 = sum(1 for _, v in brand_sorted if v >= MIN_OK)

    # độ phủ ô (hãng × aspect): bao nhiêu ô đủ MIN_OK review?
    cells_total = len(brands) * len(ASPECTS)
    cells_ok = sum(1 for b in brands for a in ASPECTS if brand_aspect[b].get(a, 0) >= MIN_OK)
    cells_any = sum(1 for b in brands for a in ASPECTS if brand_aspect[b].get(a, 0) > 0)

    # Shopee products: số review / sản phẩm
    prod_nrev = sorted(
        ((n, G.nodes[n].get("n_reviews", 0)) for n in products), key=lambda x: -x[1]
    )
    prod_ge20 = sum(1 for _, v in prod_nrev if v >= 20)

    stats = {
        "nodes_by_type": dict(by_type),
        "n_edges": G.number_of_edges(),
        "n_brands": len(brands),
        "brand_ge100": n_brand_ge100,
        "brand_ge30": n_brand_ge30,
        "top_brands": brand_sorted[:8],
        "aspect_total_corpus": aspect_total,
        "brand_aspect_cells": {"total": cells_total, "with_any": cells_any,
                               f"ge{MIN_OK}": cells_ok},
        "n_products": len(products),
        "product_ge20_reviews": prod_ge20,
        "top_products_nrev": prod_nrev[:6],
        "samsung_battery": brand_aspect.get("Samsung", {}).get("BATTERY"),
    }
    out = Path(cfg.artifacts_dir) / "kg_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== KG node theo loại ===", dict(by_type), f"| {G.number_of_edges()} cạnh")
    print(f"\n=== Hãng (brand) — tổng lượt nhắc aspect ===  ({len(brands)} hãng)")
    for b, v in brand_sorted:
        print(f"  {b:12} {v:5}  | aspect đủ≥{MIN_OK}: "
              f"{sum(1 for a in ASPECTS if brand_aspect[b].get(a,0)>=MIN_OK)}/10")
    print(f"\nHãng ≥100 lượt: {n_brand_ge100}/{len(brands)} | ≥{MIN_OK}: {n_brand_ge30}/{len(brands)}")
    print(f"Ô (hãng×aspect) đủ ≥{MIN_OK}: {cells_ok}/{cells_total} "
          f"(có ít nhất 1 review: {cells_any}/{cells_total})")
    print("\n=== Aspect — tổng toàn corpus ===")
    for a, v in sorted(aspect_total.items(), key=lambda x: -x[1]):
        print(f"  {a:12} {v:5}")
    print(f"\n=== Shopee products ===  {len(products)} sản phẩm, "
          f"{prod_ge20} có ≥20 review")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
