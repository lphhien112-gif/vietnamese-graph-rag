"""Unit test cho Knowledge Graph (no-GPU): build_kg, graph_query, product_context.

Bao gồm REGRESSION TEST cho bug grounding đã sửa: product_context không được khớp giả
do 1 token chung chung / stopword (vd câu hỏi 'camera chụp đêm' không kéo 'tai nghe chụp tai').
"""

import pandas as pd
import pytest

from vngraphrag.core import build_kg, graph_query, load_kg, product_context, save_kg

networkx = pytest.importorskip("networkx")  # KG cần networkx; skip nếu CI light chưa cài


def _toy():
    visfd = pd.DataFrame(
        {
            "comment": ["iPhone camera đẹp", "Samsung pin trâu", "Xiaomi giá rẻ"],
            "brand": ["Apple", "Samsung", "Xiaomi"],
            "parsed_labels": [
                [("CAMERA", "Positive")],
                [("BATTERY", "Positive"), ("BATTERY", "Negative")],
                [("PRICE", "Positive")],
            ],
        }
    )
    shopee = pd.DataFrame(
        {
            "comment": ["Áo hoodie form rộng đẹp", "Dép bánh mì đế cao êm"],
            "product_name": ["Áo hoodie form rộng unisex", "Dép bánh mì nam nữ đế cao"],
            "shop_name": ["ShopA", "ShopB"],
            "rating_star": [5, 4],
        }
    )
    return visfd, shopee


def test_build_kg_structure():
    G = build_kg(*_toy())
    types = {d.get("type") for _, d in G.nodes(data=True)}
    assert {"aspect", "brand", "sentiment", "shop", "product"} <= types
    # 10 aspect luôn được tạo
    assert sum(1 for _, d in G.nodes(data=True) if d.get("type") == "aspect") == 10
    # brand -> aspect edge tồn tại
    assert G.has_edge("Apple", "CAMERA")


def test_graph_query_sentiment_counts():
    G = build_kg(*_toy())
    sd = graph_query(G, "BATTERY")
    # BATTERY có cả Positive và Negative
    sentiments = {v["sentiment"] for v in sd.values()}
    assert "Positive" in sentiments and "Negative" in sentiments


def test_product_context_matches_relevant():
    G = build_kg(*_toy())
    hits = product_context(G, "Áo hoodie form rộng chất vải thế nào?")
    assert hits, "phải khớp được sản phẩm áo hoodie"
    assert "hoodie" in hits[0][0].lower()


def test_product_context_no_spurious_match():
    """REGRESSION: câu hỏi về camera KHÔNG được khớp sản phẩm Shopee thời trang
    chỉ vì trùng 1 token chung/stopword."""
    G = build_kg(*_toy())
    hits = product_context(G, "Camera điện thoại chụp đêm có tốt không?")
    assert hits == [], f"không được khớp giả, nhưng nhận: {hits}"


def test_product_context_requires_two_tokens():
    """Chỉ trùng 1 token đặc trưng -> không đủ để khớp."""
    G = build_kg(*_toy())
    # 'hoodie' (1 token) đơn lẻ không đủ
    hits = product_context(G, "cái hoodie")
    assert hits == []


def test_kg_save_load_roundtrip(tmp_path):
    G = build_kg(*_toy())
    p = tmp_path / "kg.pkl"
    save_kg(G, p)
    G2 = load_kg(p)
    assert G2.number_of_nodes() == G.number_of_nodes()
    assert G2.number_of_edges() == G.number_of_edges()
