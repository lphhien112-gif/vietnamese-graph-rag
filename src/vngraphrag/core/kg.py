"""Knowledge Graph: UIT-ViSFD (brand->aspect->sentiment) + Shopee (shop->product->aspect)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .data import ASPECTS, aspects_from_text

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
            _bump(G, asp, sn, "has_sentiment")
            if brand != "Unknown":
                _bump(G, brand, asp, "reviewed_on")

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


def product_context(G, question: str, limit: int = 3) -> list[tuple]:
    ql = str(question).lower()
    hits = []
    for n, d in G.nodes(data=True):
        if d.get("type") != "product":
            continue
        toks = [t for t in re.findall(r"\w+", n.lower()) if len(t) > 3]
        if any(t in ql for t in toks):
            hits.append((n, d.get("avg_rating", 0.0), d.get("n_reviews", 0)))
    return hits[:limit]


def save_kg(G, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)


def load_kg(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)
