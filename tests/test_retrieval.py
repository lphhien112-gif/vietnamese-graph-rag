"""Unit test cho logic retrieval & metric (no-GPU): maxsim, chuẩn hoá _nz, P@k/MRR.

Không nạp PhoBERT — chỉ kiểm tra các hàm toán học thuần với vector giả lập.
"""

import numpy as np

from vngraphrag.core import maxsim
from vngraphrag.rag.retrieval import _nz


def test_maxsim_identical_tokens_is_one():
    v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    # token query trùng hệt token doc -> max cosine = 1 mỗi token -> trung bình 1.0
    assert abs(maxsim(v, v) - 1.0) < 1e-6


def test_maxsim_orthogonal_is_zero():
    q = np.array([[1.0, 0.0]], dtype="float32")
    d = np.array([[0.0, 1.0]], dtype="float32")
    assert abs(maxsim(q, d)) < 1e-6


def test_maxsim_empty_returns_zero():
    v = np.array([[1.0, 0.0]], dtype="float32")
    assert maxsim(np.empty((0, 2), "float32"), v) == 0.0
    assert maxsim(v, np.empty((0, 2), "float32")) == 0.0


def test_maxsim_picks_best_doc_token():
    # query 1 token; doc có 1 token trùng + 1 token nhiễu -> phải lấy token trùng (=1)
    q = np.array([[1.0, 0.0]], dtype="float32")
    d = np.array([[0.0, 1.0], [1.0, 0.0]], dtype="float32")
    assert abs(maxsim(q, d) - 1.0) < 1e-6


def test_nz_normalizes_to_unit_range():
    x = np.array([1.0, 2.0, 3.0, 5.0])
    z = _nz(x)
    assert abs(z.min()) < 1e-9 and abs(z.max() - 1.0) < 1e-9


def test_nz_constant_returns_zeros():
    x = np.array([4.0, 4.0, 4.0])
    assert np.allclose(_nz(x), 0.0)


# ---- metric P@k / MRR (lấy trực tiếp từ evaluate.py để test cùng implementation) ----
def test_p_at_k_and_mrr():
    from vngraphrag.cli.evaluate import _mrr, _p_at_k

    # order: tài liệu xếp hạng; gold_sets[i] = tập aspect vàng của tài liệu i
    gold = [{"CAMERA"}, {"BATTERY"}, {"CAMERA"}, set(), {"CAMERA"}]
    order = [0, 1, 2, 3, 4]  # doc 0,2,4 liên quan CAMERA
    # P@1 = 1/1 (doc0 trúng); P@2 = 1/2; P@3 = 2/3
    assert _p_at_k(order, gold, "CAMERA", 1) == 1.0
    assert _p_at_k(order, gold, "CAMERA", 2) == 0.5
    assert abs(_p_at_k(order, gold, "CAMERA", 3) - 2 / 3) < 1e-9
    # MRR: doc đầu tiên liên quan ở hạng 1 -> 1.0
    assert _mrr(order, gold, "CAMERA") == 1.0


def test_mrr_first_relevant_at_rank3():
    from vngraphrag.cli.evaluate import _mrr

    gold = [set(), set(), {"PRICE"}]
    assert abs(_mrr([0, 1, 2], gold, "PRICE") - 1 / 3) < 1e-9


def test_mrr_none_relevant_is_zero():
    from vngraphrag.cli.evaluate import _mrr

    assert _mrr([0, 1], [set(), set()], "CAMERA") == 0.0
