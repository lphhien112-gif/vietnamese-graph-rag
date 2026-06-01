"""TN2 — Thống kê NER trên 800 review mẫu.

NER tổng quát (underthesea: PER/LOC/ORG) + BRAND nhận diện qua gazetteer thương hiệu.
Không có nhãn vàng NER cho domain TMĐT -> báo cáo theo SỐ LƯỢNG thực thể & độ phủ
(định tính), đúng như mô tả ở báo cáo (Mục TN2).

Kết quả -> artifacts/ner_stats.json (+ in thống kê).

Chạy:  python scripts/eval_ner.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import DocumentIndex, detect_brand  # noqa: E402

SAMPLE = 800


def ner_type(tag: str) -> str | None:
    # underthesea trả tag dạng B-PER/I-LOC/B-ORG... -> rút loại
    if "-" in tag:
        return tag.split("-", 1)[1]
    return None


def main() -> int:
    cfg = Config.load()
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("Chưa có index. Chạy build_index trước.")

    # 800 mẫu trải đều toàn corpus (deterministic, không random để tái lập)
    recs = index.records
    step = max(1, len(recs) // SAMPLE)
    sample = [recs[i]["raw"] for i in range(0, len(recs), step)][:SAMPLE]
    print(f"Mẫu: {len(sample)} review (bước {step} trên {len(recs)})")

    from underthesea import ner

    type_counts = Counter()         # số token thực thể theo loại
    entity_spans = Counter()        # số span (gộp B-...I-...) theo loại
    docs_with_entity = 0
    brand_counts = Counter()
    docs_with_brand = 0

    for i, text in enumerate(sample):
        try:
            tags = ner(text)
        except Exception:
            tags = []
        prev = None
        has_ent = False
        for tup in tags:
            tag = tup[-1] if isinstance(tup, (list, tuple)) else ""
            t = ner_type(str(tag))
            if t:
                type_counts[t] += 1
                has_ent = True
                if not (str(tag).startswith("I-") and prev == t):
                    entity_spans[t] += 1
                prev = t
            else:
                prev = None
        if has_ent:
            docs_with_entity += 1

        b = detect_brand(text)
        if b != "Unknown":
            brand_counts[b] += 1
            docs_with_brand += 1
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(sample)}")

    stats = {
        "n_sample": len(sample),
        "entity_token_counts": dict(type_counts),
        "entity_span_counts": dict(entity_spans),
        "docs_with_any_entity": docs_with_entity,
        "docs_with_brand": docs_with_brand,
        "brand_distribution": dict(brand_counts.most_common()),
    }
    out = Path(cfg.artifacts_dir) / "ner_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== NER (underthesea) — số SPAN thực thể theo loại ===")
    for t, c in entity_spans.most_common():
        print(f"  {t:5} : {c}")
    print(f"  review có ≥1 thực thể: {docs_with_entity}/{len(sample)}")
    print("\n=== BRAND (gazetteer) ===")
    print(f"  review nhận diện được brand: {docs_with_brand}/{len(sample)}")
    for b, c in brand_counts.most_common():
        print(f"  {b:10}: {c}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
