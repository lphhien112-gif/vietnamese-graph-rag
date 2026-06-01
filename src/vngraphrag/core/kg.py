"""Knowledge Graph: UIT-ViSFD (brand->aspect->sentiment) + Shopee (shop->product->aspect)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .data import _STOPWORDS, ASPECTS, aspects_from_text

SENT_COLOR = {"Positive": "lightgreen", "Negative": "lightcoral", "Neutral": "khaki"}


def build_kg(visfd, shopee, aspect_clf=None):
    """Dựng KG. Nếu có `aspect_clf` (BiLSTM đã train) thì dùng nó dự đoán aspect cho
    review Shopee (vốn không có nhãn) -> cạnh product->aspect chính xác hơn keyword."""
    import networkx as nx

    G = nx.DiGraph()
    for a in ASPECTS:
        G.add_node(a, type="aspect", color="lightblue")

    # UIT-ViSFD: brand -> aspect -> sentiment (gold)
    for _, row in visfd.iterrows():
        brand = row["brand"]
        if brand != "Unknown" and not G.has_node(brand):
            G.add_node(brand, type="brand", color="gold")
        for asp, sent in row["parsed_labels"]:
            if asp not in ASPECTS:
                continue
            sn = f"{asp}#{sent}"
            if not G.has_node(sn):
                G.add_node(sn, type="sentiment", sentiment=sent, color=SENT_COLOR.get(sent, "lightgray"))
            _bump(G, asp, sn, "has_sentiment")  # toàn corpus (fallback)
            if brand != "Unknown":
                _bump(G, brand, asp, "reviewed_on")
                # brand -> aspect#sentiment: cho phép truy vấn cảm xúc THEO HÃNG
                # (vd "pin Samsung?" trả thống kê riêng Samsung, không phải toàn corpus).
                _bump(G, brand, sn, "brand_sentiment")

    # Shopee: shop -> product -> aspect (mentions); product carries avg rating
    # aspect lấy từ model BiLSTM đã train (nếu có) — đây là chỗ model được DEPLOY vào KG.
    comments = [str(c) for c in shopee["comment"]] if len(shopee) else []
    if aspect_clf is not None and comments:
        shopee_aspects = aspect_clf.predict(comments)
    else:
        shopee_aspects = [aspects_from_text(c) for c in comments]

    prod_stats: dict[str, list] = {}
    for (_, r), asp_set in zip(shopee.iterrows(), shopee_aspects, strict=False):
        shop = str(r["shop_name"])[:40]
        prod = str(r["product_name"])[:50]
        if not G.has_node(shop):
            G.add_node(shop, type="shop", color="violet")
        if not G.has_node(prod):
            G.add_node(prod, type="product", color="wheat")
        _bump(G, shop, prod, "sells")
        for asp in asp_set:
            if asp in ASPECTS:
                _bump(G, prod, asp, "mentions")
        prod_stats.setdefault(prod, []).append(r.get("rating_star"))

    for prod, rs in prod_stats.items():
        vals = [float(x) for x in rs if str(x).strip() not in ("", "nan")]
        G.nodes[prod]["avg_rating"] = float(np.mean(vals)) if vals else 0.0
        G.nodes[prod]["n_reviews"] = len(rs)
    return G


def _bump(G, u, v, relation):
    if G.has_edge(u, v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, relation=relation, weight=1)


def graph_query(G, aspect: str) -> dict:
    out = {}
    if aspect in G:
        for nb in G.successors(aspect):
            d = G.nodes[nb]
            if d.get("type") == "sentiment":
                out[nb] = {"sentiment": d["sentiment"], "count": G[aspect][nb]["weight"]}
    return out


def graph_query_brand(G, brand: str, aspect: str) -> dict:
    """Phân bố cảm xúc của `aspect` CHO RIÊNG `brand` (cạnh brand->aspect#sentiment).
    Rỗng nếu hãng không có review về aspect đó -> caller tự fallback về thống kê toàn cục."""
    out = {}
    if brand in G:
        prefix = f"{aspect}#"
        for nb in G.successors(brand):
            d = G.nodes[nb]
            if d.get("type") == "sentiment" and nb.startswith(prefix):
                out[nb] = {"sentiment": d["sentiment"], "count": G[brand][nb]["weight"]}
    return out


def product_context(G, question: str, limit: int = 3) -> list[tuple]:
    """Trả về product có TÊN khớp câu hỏi. Yêu cầu >=2 token đặc trưng trùng nhau để
    tránh khớp giả do 1 từ chung chung (vd 'chụp' trong 'chụp đêm' vs 'tai nghe chụp tai').
    Xếp theo số token trùng giảm dần -> sản phẩm liên quan nhất lên trước."""
    def _toks(s):
        return {t for t in re.findall(r"\w+", s.lower()) if len(t) > 2 and t not in _STOPWORDS}

    qtokens = _toks(str(question))
    hits = []
    for n, d in G.nodes(data=True):
        if d.get("type") != "product":
            continue
        ptoks = _toks(n)
        overlap = ptoks & qtokens
        if len(overlap) >= 2:
            hits.append((len(overlap), n, d.get("avg_rating", 0.0), d.get("n_reviews", 0)))
    hits.sort(key=lambda x: -x[0])
    return [(n, r, c) for _, n, r, c in hits[:limit]]


def save_kg(G, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_kg(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
