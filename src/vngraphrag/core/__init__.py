"""Tầng core: dữ liệu + biểu diễn (data, embeddings, index, kg)."""

from .data import (
    ASPECT_KEYWORDS,
    ASPECTS,
    BRAND_GAZETTEER,
    aspect_from_query,
    aspects_from_text,
    build_records,
    detect_brand,
    load_shopee,
    load_visfd,
    parse_label,
    preprocess_vietnamese,
)
from .embeddings import PhoBERTEncoder, maxsim
from .index import DocumentIndex
from .kg import SENT_COLOR, build_kg, graph_query, load_kg, product_context, save_kg

__all__ = [
    "ASPECTS",
    "ASPECT_KEYWORDS",
    "BRAND_GAZETTEER",
    "parse_label",
    "aspects_from_text",
    "aspect_from_query",
    "detect_brand",
    "preprocess_vietnamese",
    "load_visfd",
    "load_shopee",
    "build_records",
    "PhoBERTEncoder",
    "maxsim",
    "DocumentIndex",
    "build_kg",
    "save_kg",
    "load_kg",
    "graph_query",
    "product_context",
    "SENT_COLOR",
]
