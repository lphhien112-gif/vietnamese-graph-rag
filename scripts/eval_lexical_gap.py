"""GAP 2 — Đánh giá retrieval trên bộ truy vấn CÓ LEXICAL GAP thật.

Vì sao cần: bộ 8 truy vấn của TN1 (eval_embeddings.py) GIÀU từ khóa aspect
("camera chụp ảnh đẹp không") nên TF-IDF thuần từ vựng đã gần ngang các phương pháp
ngữ nghĩa — tức là KHÔNG test được "lexical gap" (nỗi đau §1.2). Bộ này gồm các câu
hỏi đời thường KHÔNG chứa từ khóa của aspect (vd hỏi pin nhưng không có chữ "pin/sạc"),
buộc phương pháp phải hiểu NGHĨA mới truy đúng review.

Phương pháp: tái dùng nguyên 4 hàm run_* trong eval_embeddings.py, chỉ thay bộ truy
vấn (gold vẫn = nhãn aspect; pool vẫn = toàn corpus 10.923) -> so sánh CÔNG BẰNG với TN1.

TRUNG THỰC: bộ truy vấn được CHỐT trước khi chạy; báo cáo đúng số chạy ra, kể cả khi
PhoBERT/ngữ nghĩa không thắng. Mỗi câu được code kiểm tra 0 rò rỉ từ khóa aspect đích.

Chạy:  python scripts/eval_lexical_gap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_embeddings as ee  # noqa: E402  (tái dùng run_tfidf/run_glove_svd/run_word2vec/run_phobert)

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import DocumentIndex  # noqa: E402
from vngraphrag.core.data import ASPECT_KEYWORDS  # noqa: E402

# ── Bộ truy vấn lexical-gap (CHỐT TRƯỚC) — câu hỏi tự nhiên, KHÔNG dùng từ khóa aspect ──
LEXGAP_QUERIES = [
    ("dùng cả ngày trời mà vẫn chưa cần cắm điện", "BATTERY"),
    ("xài tới tối là gần cạn phải tìm ổ điện", "BATTERY"),
    ("buổi tối thiếu sáng thì hình lên có rõ không", "CAMERA"),
    ("tự sướng lên mặt có nét và sáng không", "CAMERA"),
    ("ra ngoài nắng gắt nhìn vào có thấy gì không", "SCREEN"),
    ("chạm vuốt trên mặt kính có ăn tay không", "SCREEN"),
    ("mở một lúc chục ứng dụng có bị khựng không", "PERFORMANCE"),
    ("làm việc nặng lâu lâu là thấy đơ cả máy", "PERFORMANCE"),
    ("bỏ ra ngần ấy liệu có xứng đáng không", "PRICE"),
    ("so với chất lượng thì mua có hời không", "PRICE"),
    ("nhìn ngoài đời trông sang hay trông xoàng", "DESIGN"),
    ("lưu được kha khá phim với tài liệu không", "STORAGE"),
    ("máy hỏng trong thời gian đầu có được đổi không", "SER&ACC"),
    ("người bán phản hồi tin nhắn có nhiệt tình không", "SER&ACC"),
    ("nghe nhạc xem phim âm lượng to rõ không", "FEATURES"),
    ("mở khoá bằng dấu tay có ăn và chuẩn không", "FEATURES"),
]


def verify_no_keyword_leak() -> None:
    """Khẳng định mỗi câu KHÔNG chứa từ khóa của aspect đích -> đúng nghĩa 'lexical gap'."""
    bad = []
    for q, asp in LEXGAP_QUERIES:
        leaked = [k for k in ASPECT_KEYWORDS[asp] if k in q.lower()]
        if leaked:
            bad.append((q, asp, leaked))
    if bad:
        for q, asp, lk in bad:
            print(f"  ✗ RÒ RỈ [{asp}] trong {q!r}: {lk}")
        raise SystemExit("Có câu hỏi chứa từ khóa aspect đích -> không phải lexical gap. Sửa lại.")
    print(f"✓ {len(LEXGAP_QUERIES)} câu hỏi: 0 rò rỉ từ khóa aspect đích (lexical gap thật).\n")


def main() -> int:
    verify_no_keyword_leak()

    cfg = Config.load()
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("Chưa có index. Chạy: python -m vngraphrag.cli.build_index")

    # Cùng pool & cùng định nghĩa gold với TN1 (eval_embeddings.main) để so sánh được.
    all_idx = list(range(len(index.records)))
    corpus = [index.records[i]["raw"] for i in all_idx]
    gold = [index.records[i]["gold"] for i in all_idx]

    # Thay bộ truy vấn dùng chung mà 4 hàm run_* đọc tới (module-global).
    ee.EVAL_QUERIES = LEXGAP_QUERIES

    print(f"Pool: {len(corpus)} review · {len(LEXGAP_QUERIES)} truy vấn lexical-gap\n")
    results = {}
    print("TF-IDF ...")
    results["TF-IDF"] = ee.run_tfidf(corpus, gold)
    print("GloVe-SVD ...")
    results["GloVe-SVD"] = ee.run_glove_svd(corpus, gold)
    print("Word2Vec (train SGNS) ...")
    results["Word2Vec"] = ee.run_word2vec(corpus, gold)
    print("PhoBERT ...")
    results["PhoBERT"] = ee.run_phobert(cfg, all_idx, gold)

    out = Path(cfg.artifacts_dir) / "embedding_eval_lexgap.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # In bảng + so sánh với bộ keyword-rich (TN1) nếu có.
    kw_path = Path(cfg.artifacts_dir) / "embedding_eval.json"
    kw = json.loads(kw_path.read_text(encoding="utf-8")) if kw_path.exists() else {}

    print("\n=== LEXICAL-GAP (P@5 / P@10 / MRR) ===")
    print(f"{'Method':12} {'P@5':>7} {'P@10':>7} {'MRR':>7}   | {'MRR keyword-rich (TN1)':>22}")
    for m in ["TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]:
        r = results[m]
        kwmrr = kw.get(m, {}).get("MRR")
        kwstr = f"{kwmrr:.4f}" if kwmrr is not None else "—"
        print(f"{m:12} {r['P@5']:7.4f} {r['P@10']:7.4f} {r['MRR']:7.4f}   | {kwstr:>22}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
