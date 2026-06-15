"""TN1 (mở rộng) — So sánh phương pháp biểu diễn cho RETRIEVAL, có:

  • BASELINE KEYWORD SEARCH thuần (mô phỏng ô tìm kiếm của sàn TMĐT): xếp hạng theo
    số từ trùng lặp thô giữa truy vấn và review — KHÔNG ngữ nghĩa, KHÔNG IDF. Đây là
    "việc KHÔNG dùng học biểu diễn" để đối chiếu với "việc dùng" (câu hỏi của đề bài).
  • 5 phương pháp: Keyword, TF-IDF, Word2Vec (SGNS), GloVe-SVD, PhoBERT.
  • Đánh giá trên 2 bộ truy vấn: keyword-rich (25 câu, đủ 10 aspect) và lexical-gap (16 câu).
  • KHOẢNG TIN CẬY BOOTSTRAP 95% cho MRR (resample truy vấn 2000 lần) — để biết
    chênh lệch giữa các phương pháp có vượt nhiễu thống kê hay không (bộ truy vấn nhỏ).

Train mỗi biểu diễn MỘT lần trên corpus rồi chấm cả 2 bộ truy vấn -> nhanh & công bằng.

Chạy:  python scripts/eval_embeddings_full.py
Ghi:   artifacts/embeddings_full.json
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

# Tái dùng đúng 2 bộ truy vấn đã chốt ở nơi khác (single source of truth)
from vngraphrag.cli.evaluate import EVAL_QUERIES as KEYWORD_RICH_QUERIES  # 25 câu  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from eval_lexical_gap import LEXGAP_QUERIES  # 16 câu  # noqa: E402

_TOK = re.compile(r"[a-zà-ỹ0-9]+", re.IGNORECASE)
RNG = np.random.default_rng(42)
N_BOOT = 2000


def tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(n, 1e-9, None)


# ── metrics (per-query) ───────────────────────────────────────────────────────
def _rank_metrics(order: np.ndarray, gold: list[set], asp: str) -> tuple[float, float, float]:
    p5 = sum(1 for i in order[:5] if asp in gold[i]) / 5
    p10 = sum(1 for i in order[:10] if asp in gold[i]) / 10
    rr = 0.0
    for r, i in enumerate(order, 1):
        if asp in gold[i]:
            rr = 1.0 / r
            break
    return p5, p10, rr


def per_query_scores(D: np.ndarray, Q: np.ndarray, queries, gold) -> dict:
    """D,Q đã L2-norm -> xếp hạng bằng dot. Trả per-query P@5/P@10/MRR."""
    p5s, p10s, mrrs = [], [], []
    for (_, asp), qv in zip(queries, Q, strict=False):
        order = np.argsort(D @ qv)[::-1]
        p5, p10, rr = _rank_metrics(order, gold, asp)
        p5s.append(p5)
        p10s.append(p10)
        mrrs.append(rr)
    return {"P@5": p5s, "P@10": p10s, "MRR": mrrs}


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean + khoảng tin cậy 95% bootstrap (resample truy vấn có hoàn lại)."""
    a = np.asarray(values, dtype="float64")
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    means = a[RNG.integers(0, len(a), size=(n_boot, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── (0) KEYWORD baseline — lexical overlap thô (mô phỏng search sàn TMĐT) ──────
def keyword_rank(doc_tok_sets: list[set], queries) -> dict:
    """Score doc = |Q ∩ doc tokens|. Không IDF, không vector — đúng kiểu tìm từ khoá."""
    def scores_for(q: str) -> np.ndarray:
        qs = set(tok(q))
        return np.array([len(qs & d) for d in doc_tok_sets], dtype="float32")

    res = {"P@5": [], "P@10": [], "MRR": []}
    for q, asp in queries:
        order = np.argsort(scores_for(q))[::-1]
        p5, p10, rr = _rank_metrics(order, gold_global, asp)
        res["P@5"].append(p5)
        res["P@10"].append(p10)
        res["MRR"].append(rr)
    return res


# ── (1) TF-IDF (sparse, L2-norm sẵn) ──────────────────────────────────────────
def tfidf_repr(corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(tokenizer=tok, token_pattern=None, min_df=2, max_features=20000)
    D = vec.fit_transform(corpus)  # rows L2-norm mặc định
    return vec, D


def tfidf_scores(vec, D, queries, gold):
    res = {"P@5": [], "P@10": [], "MRR": []}
    for q, asp in queries:
        qv = vec.transform([q])
        sims = (D @ qv.T).toarray().ravel()
        order = np.argsort(sims)[::-1]
        p5, p10, rr = _rank_metrics(order, gold, asp)
        res["P@5"].append(p5)
        res["P@10"].append(p10)
        res["MRR"].append(rr)
    return res


# ── (2) GloVe-SVD (PPMI co-occurrence -> SVD) ─────────────────────────────────
def glove_svd_repr(corpus, dim=100, window=5, vocab_size=6000):
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
    rows, cols, vals = zip(*[(i, j, c) for (i, j), c in co.items()], strict=False)
    C = csr_matrix((vals, (rows, cols)), shape=(V, V), dtype="float64")
    total = C.sum()
    rowsum = np.asarray(C.sum(1)).ravel()
    colsum = np.asarray(C.sum(0)).ravel()
    C = C.tocoo()
    pmi = np.log((C.data * total) / (rowsum[C.row] * colsum[C.col]) + 1e-12)
    pmi[pmi < 0] = 0.0
    P = csr_matrix((pmi, (C.row, C.col)), shape=(V, V))
    W = l2(TruncatedSVD(n_components=dim, random_state=42).fit_transform(P)).astype("float32")
    return stoi, W, dim, toks


def _embed_avg(words, stoi, W, dim):
    ids = [stoi[w] for w in words if w in stoi]
    return W[ids].mean(0) if ids else np.zeros(dim, "float32")


# ── (3) Word2Vec (skip-gram + negative sampling, PyTorch) ─────────────────────
def word2vec_repr(corpus, dim=100, window=5, neg=5, epochs=3, vocab_size=8000):
    from collections import Counter

    import torch

    toks = [tok(c) for c in corpus]
    freq = Counter(w for t in toks for w in t)
    itos = [w for w, c in freq.most_common(vocab_size) if c >= 2]
    stoi = {w: i for i, w in enumerate(itos)}
    V = len(itos)
    centers, contexts = [], []
    for t in toks:
        ids = [stoi[w] for w in t if w in stoi]
        for a in range(len(ids)):
            for b in range(max(0, a - window), min(len(ids), a + window + 1)):
                if a != b:
                    centers.append(ids[a])
                    contexts.append(ids[b])
    centers = torch.tensor(centers)
    contexts = torch.tensor(contexts)
    f = np.array([freq[w] for w in itos], dtype="float64") ** 0.75
    negp = torch.tensor(f / f.sum())
    torch.manual_seed(42)
    Win = torch.nn.Embedding(V, dim)
    Wout = torch.nn.Embedding(V, dim)
    torch.nn.init.normal_(Win.weight, std=0.1)
    torch.nn.init.normal_(Wout.weight, std=0.1)
    opt = torch.optim.Adam(list(Win.parameters()) + list(Wout.parameters()), lr=2e-3)
    bs, N = 4096, len(centers)
    for ep in range(epochs):
        perm = torch.randperm(N)
        tot = 0.0
        for i in range(0, N, bs):
            idx = perm[i : i + bs]
            c, o = centers[idx], contexts[idx]
            n = torch.multinomial(negp, len(idx) * neg, replacement=True).view(len(idx), neg)
            vc, vo, vn = Win(c), Wout(o), Wout(n)
            pos = torch.nn.functional.logsigmoid((vc * vo).sum(1))
            negs = torch.nn.functional.logsigmoid(-(vn * vc.unsqueeze(1)).sum(2)).sum(1)
            loss = -(pos + negs).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"    [w2v] epoch {ep + 1}/{epochs} loss={tot / N:.4f}")
    W = l2(Win.weight.detach().numpy().astype("float32"))
    return stoi, W, dim, toks


def main() -> int:
    global gold_global
    cfg = Config.load()
    index = DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
    if index is None:
        raise SystemExit("Chưa có index. Chạy: python -m vngraphrag.cli.build_index")

    corpus = [r["raw"] for r in index.records]
    gold = [r["gold"] for r in index.records]
    gold_global = gold
    n_uit = sum(1 for r in index.records if r.get("source") == "UIT-ViSFD")
    qsets = {"keyword_rich": list(KEYWORD_RICH_QUERIES), "lexical_gap": list(LEXGAP_QUERIES)}
    print(f"Pool: {len(corpus)} review ({n_uit} UIT gold + {len(corpus) - n_uit} Shopee nhiễu)")
    print(f"Query sets: keyword_rich={len(qsets['keyword_rich'])}  lexical_gap={len(qsets['lexical_gap'])}\n")

    # Fit MỖI biểu diễn 1 lần ----------------------------------------------------
    print("Keyword baseline (lexical overlap)...")
    doc_tok_sets = [set(tok(c)) for c in corpus]

    print("TF-IDF (fit)...")
    tfv, tfD = tfidf_repr(corpus)

    print("GloVe-SVD (train)...")
    g_stoi, g_W, g_dim, g_toks = glove_svd_repr(corpus)
    gD = l2(np.vstack([_embed_avg(t, g_stoi, g_W, g_dim) for t in g_toks]))

    print("Word2Vec (train SGNS)...")
    w_stoi, w_W, w_dim, w_toks = word2vec_repr(corpus)
    wD = l2(np.vstack([_embed_avg(t, w_stoi, w_W, w_dim) for t in w_toks]))

    print("PhoBERT (query encode; doc vectors đã cache)...")
    from vngraphrag.core import PhoBERTEncoder

    phoD = l2(index.vectors.astype("float32"))
    enc = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)

    # Chấm cả 2 bộ truy vấn ------------------------------------------------------
    out: dict = {"pool_size": len(corpus), "n_boot": N_BOOT, "sets": {}}
    for sname, queries in qsets.items():
        print(f"\n=== Query set: {sname} ({len(queries)} câu) ===")
        gQ = l2(np.vstack([_embed_avg(tok(q), g_stoi, g_W, g_dim) for q, _ in queries]))
        wQ = l2(np.vstack([_embed_avg(tok(q), w_stoi, w_W, w_dim) for q, _ in queries]))
        pQ = l2(enc.encode_mean([q for q, _ in queries]).astype("float32"))

        methods = {
            "Keyword": keyword_rank(doc_tok_sets, queries),
            "TF-IDF": tfidf_scores(tfv, tfD, queries, gold),
            "Word2Vec": per_query_scores(wD, wQ, queries, gold),
            "GloVe-SVD": per_query_scores(gD, gQ, queries, gold),
            "PhoBERT": per_query_scores(phoD, pQ, queries, gold),
        }

        set_res = {}
        print(f"{'Method':10} {'P@5':>6} {'P@10':>6} {'MRR':>6}   {'MRR 95% CI':>18}")
        for m in ["Keyword", "TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]:
            r = methods[m]
            mrr_mean, lo, hi = bootstrap_ci(r["MRR"])
            set_res[m] = {
                "P@5": round(float(np.mean(r["P@5"])), 4),
                "P@10": round(float(np.mean(r["P@10"])), 4),
                "MRR": round(mrr_mean, 4),
                "MRR_ci95": [round(lo, 4), round(hi, 4)],
                "MRR_per_query": [round(x, 4) for x in r["MRR"]],
            }
            print(f"{m:10} {set_res[m]['P@5']:6.3f} {set_res[m]['P@10']:6.3f} "
                  f"{set_res[m]['MRR']:6.3f}   [{lo:.3f}, {hi:.3f}]")
        out["sets"][sname] = set_res

    op = Path(cfg.artifacts_dir) / "embeddings_full.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
