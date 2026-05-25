# Kế hoạch nhóm — Vietnamese Graph RAG

## 1. Trạng thái hiện tại
| Hạng mục | Trạng thái |
|---|---|
| Notebook Part 1 (data · NER · embedding · Subword · BiLSTM) | ✅ code xong, chờ chạy lấy số |
| Notebook Part 2 (KG · MaxSim · RAG · export · Gradio) | ✅ code xong, chờ chạy lấy số |
| Package LLMOps `src/vngraphrag` (+ API, UI, CLI, CI, Docker) | ✅ code xong, `make check` 0 lỗi |
| Train→deploy BiLSTM (`make train` / notebook §6 → `/classify` + KG) | ✅ |
| Báo cáo `report/NLP_Final_Report.tex` | ✅ đủ section, **chờ điền số liệu + hình thật** |
| 2 dataset (UIT-ViSFD + Shopee) | ✅ trong `data/raw/` |

## 2. Việc còn lại (BẮT BUỘC chạy mới có số thật)
1. Chạy **Part 1** trên Kaggle (Internet + GPU) → lấy: bảng P@k/MRR, micro-F1 BiLSTM, hình `fig_embedding_comparison.png`, `results_part1.json`.
2. Chạy **Part 2** (thêm Secret `OPENAI_API_KEY`) → lấy: bảng ablation, `fig_knowledge_graph.png`, `fig_rag_vs_graphrag.png`, `results_part2.json`, folder `artifacts/`.
3. **Điền số vào báo cáo** (các chỗ `\dots` / `[điền...]`):
   - Thí nghiệm 1 ← `results_part1.json`
   - Thí nghiệm 2b (micro-F1) ← `results_part1.json`
   - Thí nghiệm 3 (ablation) ← `results_part2.json`
4. **Copy hình** vào `report/figures/` (đè bản cũ): `embedding_comparison.png`, `knowledge_graph.png`, `rag_vs_graphrag.png`.
5. Điền **tên giảng viên** ở trang bìa; build lại PDF (`pdflatex` x2).
6. (Tùy chọn) serve LLMOps: tải `artifacts/` về → `import_artifacts` → `uvicorn app.api:app` + `app.ui`.

## 3. Phân công
| Hiên (NT) | Đình | Huyên |
|---|---|---|
| Kiến trúc, RAG pipeline, **LLMOps** (API/CI/Docker/observability), tích hợp | Tiền xử lý, NER, **RNN/BiLSTM train→deploy**, dataset **Shopee** | Embedding, **Subword**, Knowledge Graph, **Gradio UI**, evaluation |
| Báo cáo: tổng thể | Báo cáo: Data/NER/RNN | Báo cáo: Embedding/KG |

## 4. Lệnh chạy nhanh (repo, Windows PowerShell)
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
$env:OPENAI_API_KEY = "sk-..."
python scripts\check_repo.py            # kiểm tra repo
python -m vngraphrag.cli.train_aspect   # train BiLSTM (GPU)
python -m vngraphrag.cli.build_index    # index + KG
python -m vngraphrag.cli.evaluate       # P@k/MRR + F1 + gate
uvicorn app.api:app --port 8000         # API ; hoặc: python -m app.ui
```
Máy yếu → train/index trên Kaggle, tải `artifacts/` về, `import_artifacts` rồi chỉ serve.
