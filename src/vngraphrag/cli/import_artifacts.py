"""CLI: kiểm tra & xác nhận folder artifacts/ (xuất từ notebook Part 2) hợp lệ để serve.

python -m vngraphrag.cli.import_artifacts
python -m vngraphrag.cli.import_artifacts --artifacts-dir ./artifacts
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from ..config import Config
from ..core import DocumentIndex

REQUIRED = ["doc_vectors.npy", "records.json", "manifest.json", "kg.pkl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--artifacts-dir", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    d = Path(args.artifacts_dir or cfg.artifacts_dir)

    missing = [f for f in REQUIRED if not (d / f).exists()]
    if missing:
        raise SystemExit(f"❌ Thiếu file {missing} trong {d}/  (chạy cell §7 notebook Part 2 rồi tải artifacts/ về)")

    idx = DocumentIndex.load(d, cfg.embedding_model)
    if idx is None:
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        raise SystemExit(
            f"❌ Version lệch: manifest model={man.get('model')} / version={man.get('version')} "
            f"nhưng config embedding_model={cfg.embedding_model}.\n"
            f"   -> Đảm bảo notebook và config.yaml dùng CÙNG model."
        )

    if len(idx.vectors) != len(idx.records):
        raise SystemExit(f"❌ Lệch: {len(idx.vectors)} vectors != {len(idx.records)} records")

    with open(d / "kg.pkl", "rb") as f:
        G = pickle.load(f)
    src = Counter(r.get("source") for r in idx.records)

    has_clf = (d / "aspect_clf.pt").exists()
    print("✅ artifacts hợp lệ:")
    print(f"   index : {len(idx.records)} records · dim={idx.vectors.shape[1]} · version={idx.version}")
    print(f"   nguồn : {dict(src)}")
    print(f"   KG    : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(
        f"   model : aspect_clf.pt {'✅ (deploy /classify)' if has_clf else '— chưa có (/classify fallback keyword)'}"
    )
    print("   → sẵn sàng serve: make api   (không cần encode lại corpus)")


if __name__ == "__main__":
    main()
