# Vietnamese Graph RAG for E-commerce 🛒
### Hệ thống Hỏi đáp Thông minh cho Thương mại Điện tử Tiếng Việt

> NLP Final Project — ĐH Khoa học Tự nhiên, ĐHQG-HCM

## 👥 Nhóm thực hiện
| Thành viên | MSSV | Vai trò |
|---|---|---|
| Lê Phạm Hồng Hiên | 22110059 | Nhóm trưởng — Kiến trúc, RAG Pipeline |
| Dương Bùi Phương Đình | 22110049 | Data Engineering, NER |
| Phạm Xuân Huyên | 22110076 | Embedding, Knowledge Graph |

## 📋 Mô tả dự án

Xây dựng hệ thống **Graph RAG** (Retrieval-Augmented Generation) cho domain **Thương mại Điện tử tiếng Việt**. Hệ thống cho phép người dùng đặt câu hỏi về sản phẩm (ví dụ: *"Camera iPhone 15 chụp đêm có tốt không?"*) và trả lời dựa trên đánh giá thực tế từ Shopee/Lazada, kết hợp Knowledge Graph để tăng chất lượng truy xuất.

### Kiến thức NLP áp dụng
- **Week 1**: TF-IDF — Baseline retrieval
- **Week 2**: Word2Vec — Document embedding
- **Week 3**: GloVe — Co-occurrence embedding  
- **Week 4**: Attention — Re-ranking documents
- **Week 5**: PhoBERT/Transformer — SOTA embedding + NER

## 📦 Dataset

| Dataset | Kích thước | Nội dung | Nguồn |
|---|---|---|---|
| **UIT-ViSFD** | 11,122 comments | Review smartphone, 10 aspects, 3 sentiments | Shopee |
| **ViCloABSA** | 7,000 comments | Review quần áo, 5 aspects | Shopee + Lazada |
| **PhoNER_COVID19** | 10 entity types | NER tiếng Việt | HuggingFace |

## 🏗️ Cấu trúc dự án
```
vietnamese-graph-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      ← Đình
│   ├── 02_embedding_comparison.ipynb    ← Huyên
│   ├── 03_ner_extraction.ipynb          ← Đình
│   ├── 04_knowledge_graph.ipynb         ← Huyên
│   ├── 05_retrieval_evaluation.ipynb    ← Hiên
│   └── 06_full_pipeline_demo.ipynb      ← Hiên
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── ner_model.py
│   ├── knowledge_graph.py
│   ├── retrieval.py
│   └── rag_pipeline.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/figures/
└── report/
    ├── NLP_Final_Report.tex
    └── figures/
```

## 🚀 Cài đặt
```bash
git clone https://github.com/<username>/vietnamese-graph-rag.git
cd vietnamese-graph-rag
pip install -r requirements.txt
```

## 📄 License
Academic use only — NLP Final Project 2025-2026.
