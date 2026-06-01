"""Sinh 2 hình cho báo cáo từ số liệu thật, đảm bảo khớp bảng trong report:
  report/figures/embedding_comparison.png  <- artifacts/embedding_eval.json
  report/figures/knowledge_graph.png       <- copy từ artifacts/kg_brand_aspect.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ── embedding_comparison.png ─────────────────────────────────────────────────
res = json.loads((ART / "embedding_eval.json").read_text(encoding="utf-8"))
methods = ["TF-IDF", "Word2Vec", "GloVe-SVD", "PhoBERT"]
metrics = ["P@5", "P@10", "MRR"]
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(methods))
w = 0.26
colors = ["#6BAED6", "#74C476", "#FD8D3C"]
for j, m in enumerate(metrics):
    vals = [res[me][m] for me in methods]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=m, color=colors[j])
    for b, v in zip(bars, vals, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0.7, 0.92)
ax.set_ylabel("Điểm")
ax.set_title("So sánh hiệu quả Retrieval — 4 phương pháp Embedding (pool 10.923 review, 8 truy vấn)")
ax.legend(title="Độ đo")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "embedding_comparison.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ── knowledge_graph.png ──────────────────────────────────────────────────────
src = ART / "kg_brand_aspect.png"
if src.exists():
    shutil.copy(src, FIG / "knowledge_graph.png")

print("✅ Đã sinh:")
print("  ", FIG / "embedding_comparison.png")
print("  ", FIG / "knowledge_graph.png")
