"""TN1 — So sánh 4 phương pháp embedding cho RETRIEVAL trên UIT-ViSFD.

Cùng 1 task & cùng phán quyết liên quan (nhãn vàng aspect) cho cả 4 phương pháp:
  TF-IDF        : sklearn TfidfVectorizer
  Word2Vec      : skip-gram + negative sampling (tự train bằng PyTorch — gensim không
                  build được trên Python 3.14). doc/query = mean word vectors.
  GloVe-SVD     : PPMI co-occurrence -> TruncatedSVD (giống GloVe qua SVD ma trận đồng xuất hiện)
  PhoBERT       : vector mean-pool đã encode sẵn (artifacts/doc_vectors.npy)

Metric: Precision@5, Precision@10, MRR — trung bình trên bộ truy vấn (cùng bộ với TN3).
Kết quả -> artifacts/embedding_eval.json (+ in bảng).

Chạy:  python scripts/eval_embeddings.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vngraphrag.config import Config  # noqa: E402
from vngraphrag.core import DocumentIndex  # noqa: E402

# Cùng bộ truy vấn với evaluate.py (TN3) để 2 bảng so sánh được với nhau.
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

_TOK = re.compile(r"[a-zà-ỹ0-9]+", re.IGNORECASE)


def tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


# ── metrics ───────────────────────────────────────────────────────────────────
def p_at_k(order, gold, asp, k):
    return sum(1 for i in order[:k] if asp in gold[i]) / k


def mrr(order, gold, asp):
    for r, i in enumerate(order, 1):
        if asp in gold[i]:
            return 1.0 / r
    return 0.0


def score(doc_vecs, q_vecs, gold):
    """doc_vecs (N,d) & q_vecs (Q,d) đã chuẩn hoá L2 -> xếp hạng bằng dot product."""
    p5 = p10 = mr = 0.0
    for (_, asp), qv in zip(EVAL_QUERIES, q_vecs, strict=False):
        sims = doc_vecs @ qv
        order = np.argsort(sims)[::-1]
        p5 += p_at_k(order, gold, asp, 5)
        p10 += p_at_k(order, gold, asp, 10)
        mr += mrr(order, gold, asp)
    n = len(EVAL_QUERIES)
    return {"P@5": round(p5 / n, 4), "P@10": round(p10 / n, 4), "MRR": round(mr / n, 4)}


def l2(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


# ── TF-IDF ────────────────────────────────────────────────────────────────────
def run_tfidf(corpus, gold):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, min_df=2, max_features=20000)
    D = vec.fit_transform(corpus)  # sparse (N, V)
    Q = vec.transform([q for q, _ in EVAL_QUERIES])
    D = l2(np.asarray(D.todense())).astype("float32")
    Q = l2(np.asarray(Q.todense())).astype("float32")
    return score(D, Q, gold)


# ── GloVe-SVD (PPMI co-occurrence -> SVD) ─────────────────────────────────────
def run_glove_svd(corpus, gold, dim=100, window=5, vocab_size=6000):
    from collections import Counter

    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    toks = [tok(c) for c in corpus]
    freq = Counter(w for t in toks for w in t)
    itos = [w for w, _ in freq.most_common(vocab_size)]
    stoi = {w: i for i, w in enumerate(itos)}
    V = len(itos)
    co = Counter()
    for t in toks:
        ids = [stoi[w] for w in t if w in stoi]
        for a in range(len(ids)):
            for b in range(max(0, a - window), min(len(ids), a + window + 1)):
                if a != b:
                    co[(ids[a], ids[b])] += 1
    if not co:
        return {"P@5": 0.0, "P@10": 0.0, "MRR": 0.0}
    rows, cols, vals = zip(*[(i, j, c) for (i, j), c in co.items()], strict=False)
    C = csr_matrix((vals, (rows, cols)), shape=(V, V), dtype="float64")
    # PPMI
    total = C.sum()
    rowsum = np.asarray(C.sum(1)).ravel()
    colsum = np.asarray(C.sum(0)).ravel()
    C = C.tocoo()
    pmi_data = np.log((C.data * total) / (rowsum[C.row] * colsum[C.col]) + 1e-12)
    pmi_data[pmi_data < 0] = 0.0
    P = csr_matrix((pmi_data, (C.row, C.col)), shape=(V, V))
    svd = TruncatedSVD(n_components=dim, random_state=42)
    W = l2(svd.fit_transform(P)).astype("float32")  # (V, dim)

    def embed(words):
        ids = [stoi[w] for w in words if w in stoi]
        return W[ids].mean(0) if ids else np.zeros(dim, "float32")

    D = l2(np.vstack([embed(t) for t in toks]))
    Q = l2(np.vstack([embed(tok(q)) for q, _ in EVAL_QUERIES]))
    return score(D, Q, gold)


# ── Word2Vec (skip-gram + negative sampling, PyTorch) ─────────────────────────
def run_word2vec(corpus, gold, dim=100, window=5, neg=5, epochs=3, vocab_size=8000):
    from collections import Counter

    import torch

    toks = [tok(c) for c in corpus]
    freq = Counter(w for t in toks for w in t)
    itos = [w for w, c in freq.most_common(vocab_size) if c >= 2]
    stoi = {w: i for i, w in enumerate(itos)}
    V = len(itos)
    if V == 0:
        return {"P@5": 0.0, "P@10": 0.0, "MRR": 0.0}

    # cặp (center, context)
    centers, contexts = [], []
    for t in toks:
        ids = [stoi[w] for w in t if w in stoi]
        for a in range(len(ids)):
            for b in range(max(0, a - window), min(len(ids), a + window + 1)):
                if a != b:
                    centers.append(ids[a])
                    contexts.append(ids[b])
    if not centers:
        return {"P@5": 0.0, "P@10": 0.0, "MRR": 0.0}
    centers = torch.tensor(centers)
    contexts = torch.tensor(contexts)

    # phân phối negative sampling ~ freq^0.75
    f = np.array([freq[w] for w in itos], dtype="float64") ** 0.75
    negp = torch.tensor(f / f.sum())

    torch.manual_seed(42)
    Win = torch.nn.Embedding(V, dim)
    Wout = torch.nn.Embedding(V, dim)
    torch.nn.init.normal_(Win.weight, std=0.1)
    torch.nn.init.normal_(Wout.weight, std=0.1)
    opt = torch.optim.Adam(list(Win.parameters()) + list(Wout.parameters()), lr=2e-3)
    bs = 4096
    N = len(centers)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            c = centers[idx]
            o = contexts[idx]
            n = torch.multinomial(negp, len(idx) * neg, replacement=True).view(len(idx), neg)
            vc = Win(c)  # (B, d)
            vo = Wout(o)  # (B, d)
            vn = Wout(n)  # (B, neg, d)
            pos = torch.nn.functional.logsigmoid((vc * vo).sum(1))
            negs = torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)
            loss = -(pos + negs).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"    [w2v] epoch {ep + 1}/{epochs} loss={tot / N:.4f}")
    W = l2(Win.weight.detach().numpy().astype("float32"))

    def embed(words):
        ids = [stoi[w] for w in words if w in stoi]
        return W[ids].mean(0) if ids else np.zeros(dim, "float32")

    D = l2(np.vstack([embed(t) for t in toks]))
    Q = l2(np.vstack([embed(tok(q)) for q, _ in EVAL_QUERIES]))
    return score(D, Q, gold)


# ── PhoBERT (dùng vector đã encode) ───────────────────────────────────────────
def run_phobert(cfg, uit_idx, gold):
    from vngraphrag.core import PhoBERTEncoder

    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    D = l2(index.vectors[uit_idx].astype("float32"))
    enc = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
    Q = l2(enc.encode_mean([q for q, _ in EVAL_QUERIES]).astype("float32"))
    return score(D, Q, gold)


def main() -> int:
    cfg = Config.load()
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("Chưa có index. Chạy: python -m vngraphrag.cli.build_index")
    # Pool xếp hạng = TOÀN corpus: UIT-ViSFD (có gold) + Shopee (gold rỗng = nhiễu không
    # liên quan) -> task phân biệt rõ hơn, sát điều kiện TN3.
    all_idx = list(range(len(index.records)))
    corpus = [index.records[i]["raw"] for i in all_idx]
    gold = [index.records[i]["gold"] for i in all_idx]
    n_uit = sum(1 for r in index.records if r.get("source") == "UIT-ViSFD")
    print(f"Pool: {len(corpus)} review ({n_uit} UIT có gold + {len(corpus) - n_uit} Shopee nhiễu) · "
          f"{len(EVAL_QUERIES)} truy vấn\n")

    results = {}
    print("TF-IDF ...")
    results["TF-IDF"] = run_tfidf(corpus, gold)
    print("GloVe-SVD ...")
    results["GloVe-SVD"] = run_glove_svd(corpus, gold)
    print("Word2Vec (train SGNS) ...")
    results["Word2Vec"] = run_word2vec(corpus, gold)
    print("PhoBERT ...")
    results["PhoBERT"] = run_phobert(cfg, all_idx, gold)

    out = Path(cfg.artifacts_dir) / "embedding_eval.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== KẾT QUẢ (P@5 / P@10 / MRR) ===")
    print(f"{'Method':12} {'P@5':>7} {'P@10':>7} {'MRR':>7}")
    for m in ["TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]:
        r = results[m]
        print(f"{m:12} {r['P@5']:7.4f} {r['P@10']:7.4f} {r['MRR']:7.4f}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
