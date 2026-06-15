"""Unit test cho data layer (no-GPU): parse_label edge cases, keyword mapping, build_records.

Đặc biệt bảo vệ bug #1 (regex làm mất aspect SER&ACC) và các nhánh khó của parse_label.
"""

import pandas as pd

from vngraphrag.core import (
    ASPECTS,
    aspect_from_query,
    aspects_from_text,
    build_records,
    detect_brand,
    parse_label,
)


def test_parse_label_serracc_not_dropped():
    """REGRESSION bug #1: ký tự '&' trong SER&ACC phải được giữ."""
    pairs = parse_label("{SER&ACC#Positive};{CAMERA#Negative};")
    assert ("SER&ACC", "Positive") in pairs


def test_parse_label_empty_and_others():
    assert parse_label("") == []
    assert parse_label("{OTHERS};") == []  # OTHERS không phải aspect hợp lệ


def test_parse_label_multiple():
    pairs = parse_label("{CAMERA#Positive};{BATTERY#Negative};{SCREEN#Neutral};")
    assert len(pairs) == 3
    assert ("BATTERY", "Negative") in pairs


def test_all_aspects_are_valid_labels():
    # mọi aspect trong ASPECTS phải parse được khi đặt trong nhãn
    for a in ASPECTS:
        pairs = parse_label(f"{{{a}#Positive}};")
        assert (a, "Positive") in pairs, f"{a} không parse được"


def test_aspect_from_query_returns_none_for_irrelevant():
    assert aspect_from_query("xin chào bạn khỏe không") is None


def test_aspects_from_text_multi():
    found = aspects_from_text("pin yếu, màn hình đẹp, giá rẻ")
    assert {"BATTERY", "SCREEN", "PRICE"} <= found


def test_detect_brand_gazetteer_variants():
    assert detect_brand("con redmi note này ngon") == "Xiaomi"  # redmi -> Xiaomi
    assert detect_brand("galaxy s24 đẹp") == "Samsung"  # galaxy -> Samsung
    assert detect_brand("oppo reno") == "OPPO"


def test_num_regex_keeps_alphanumeric():
    """M1 fix: chỉ bỏ SỐ ĐỨNG RIÊNG, GIỮ token chữ-số mang nghĩa (64gb/120hz/5g)."""
    from vngraphrag.core.data import _NUM_RE

    toks = _NUM_RE.sub(" ", "64gb 13 120hz 5g iphone 13 pro").split()
    assert "64gb" in toks and "120hz" in toks and "5g" in toks  # giữ token chữ-số
    assert "13" not in toks  # số đứng một mình bị bỏ


def test_build_records_shape_and_gold():
    visfd = pd.DataFrame(
        {
            "comment": ["camera đẹp pin trâu lắm luôn nhé"],
            "label": ["{CAMERA#Positive};{BATTERY#Positive};"],
        }
    )
    visfd["parsed_labels"] = visfd["label"].apply(parse_label)
    visfd["aspects"] = visfd["parsed_labels"].apply(lambda x: [a for a, _ in x])
    shopee = pd.DataFrame(
        {"comment": ["áo đẹp"], "product_name": ["Áo thun"], "shop_name": ["S"], "rating_star": [5]}
    )
    recs = build_records(visfd, shopee)
    uit = [r for r in recs if r["source"] == "UIT-ViSFD"]
    shop = [r for r in recs if r["source"] == "Shopee"]
    assert len(uit) == 1 and len(shop) == 1
    assert uit[0]["gold"] == {"CAMERA", "BATTERY"}
    assert shop[0]["gold"] == set()  # Shopee không có nhãn vàng
