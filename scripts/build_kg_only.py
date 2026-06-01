"""Build & lưu CHỈ Knowledge Graph (không encode PhoBERT — nhẹ, chạy vài giây).

KG: UIT-ViSFD (brand->aspect->sentiment, nhãn vàng) + Shopee (shop->product->aspect).
Aspect cho review Shopee: dùng BiLSTM nếu có artifacts/aspect_clf.pt, ngược lại fallback keyword.

Chạy:  python scripts/build_kg_only.py     (từ thư mục gốc repo)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import build_kg, load_shopee, load_visfd, save_kg  # noqa: E402
from vngraphrag.core.aspect_clf import AspectClassifier  # noqa: E402


def main() -> int:
    cfg = Config.load()
    print(f"[data] đọc UIT-ViSFD + Shopee từ {cfg.data_dir}/ ...")
    visfd = load_visfd(cfg.data_dir)
    shopee = load_shopee(cfg.data_dir)
    print(f"[data] UIT-ViSFD = {len(visfd)} dòng · Shopee = {len(shopee)} dòng")

    aspect_clf = AspectClassifier.load(cfg.artifacts_dir, cfg.aspect_clf_threshold)
    print(f"[aspect] BiLSTM {'ĐÃ nạp' if aspect_clf else 'CHƯA có -> fallback keyword'}")

    kg = build_kg(visfd, shopee, aspect_clf)
    out = Path(cfg.artifacts_dir) / "kg.pkl"
    save_kg(kg, out)

    # thống kê nhanh theo loại node
    from collections import Counter

    types = Counter(d.get("type", "?") for _, d in kg.nodes(data=True))
    print(f"\n✅ Saved KG -> {out}")
    print(f"   nodes = {kg.number_of_nodes()} · edges = {kg.number_of_edges()}")
    print("   phân loại node: " + " · ".join(f"{k}={v}" for k, v in sorted(types.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
