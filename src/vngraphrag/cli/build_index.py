"""CLI: build & persist the document index + Knowledge Graph.

python -m vngraphrag.cli.build_index            # build if missing/stale
python -m vngraphrag.cli.build_index --rebuild  # force rebuild
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config
from ..core import DocumentIndex, build_kg, build_records, load_shopee, load_visfd, save_kg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    from ..core import PhoBERTEncoder

    encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)

    visfd, shopee = load_visfd(cfg.data_dir), load_shopee(cfg.data_dir)
    records = build_records(visfd, shopee)
    print(f"Records: {len(records)} (UIT-ViSFD + Shopee)")

    existing = None if args.rebuild else DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if existing is None:
        print("Building index (PhoBERT encode)...")
        index = DocumentIndex.build(records, encoder, cfg.embedding_model)
        index.save(cfg.artifacts_dir)
        print(f"Saved index v{index.version} -> {cfg.artifacts_dir}")
    else:
        print(f"Index up-to-date (v{existing.version}); use --rebuild to force.")

    from ..core.aspect_clf import AspectClassifier

    aspect_clf = AspectClassifier.load(cfg.artifacts_dir, cfg.aspect_clf_threshold)
    if aspect_clf is not None:
        print("Aspect classifier deployed -> dùng để gán aspect cho review Shopee.")
    kg = build_kg(visfd, shopee, aspect_clf)
    save_kg(kg, Path(cfg.artifacts_dir) / "kg.pkl")
    print(f"Saved KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")


if __name__ == "__main__":
    main()
