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

### Bám theo 6 lecture của môn (NLP_Introduction roadmap)
| Lecture | Nội dung | Áp dụng trong project |
|---|---|---|
| Lec01 — Introduction | NLP/ML là gì | Đặt vấn đề Graph RAG |
| **Lec02 — Word Representation** | one-hot, TF-IDF, Word2Vec, GloVe | Part 1 §5.1–5.3: TF-IDF · Word2Vec · GloVe-SVD |
| **Lec03 — Subword Models** | BPE, subword | Part 1 §5.4: BPE tokenization (PhoBERT) |
| **Lec04 — Compositional Semantics** | Document→Sentence→Phrase→Word | Knowledge Graph (brand→aspect→sentiment, shop→product) |
| **Lec05 — Recurrent Neural Networks** | RNN/LSTM, dữ liệu tuần tự | Part 1 §6: BiLSTM phân loại aspect (micro-F1) |
| **Lec06 — Attention & Transformer** | attention, encoder–decoder | Part 1 §5.5 PhoBERT · Part 2 §4 MaxSim re-ranking |

→ Toàn bộ nuôi pipeline **Graph RAG** (Part 2): retrieve → graph context → sinh câu trả lời (OpenAI).

## 📦 Dataset

| Dataset | Kích thước | Nội dung | Nguồn |
|---|---|---|---|
| **UIT-ViSFD** | 11,122 comments (train 7,786) | Review smartphone, **10 aspects** (gồm SER&ACC), 3 sentiments | Shopee |
| **Shopee reviews** | 3,154 reviews · 26 sản phẩm · 24 shop | Đa sản phẩm (thời trang…), có `comment / product_name / shop_name / rating_star` | [nhtlongcs](https://github.com/nhtlongcs/shopee-reviews-sentiment-analysis) |
| **PhoNER_COVID19** | — | NER tiếng Việt (domain COVID — tải về tham khảo, **không dùng** để gán nhãn sản phẩm) | HuggingFace |

> Part 2 **gộp 2 nguồn** thành corpus ~10,900 review: UIT-ViSFD (có nhãn aspect) + Shopee (shop/product/rating cho Knowledge Graph). Cuối Part 2 có **giao diện web Gradio** (`app.launch(share=True)` → link `*.gradio.live` để demo trực tiếp).

> Ghi chú: 10 aspect = `SCREEN, CAMERA, BATTERY, PERFORMANCE, STORAGE, DESIGN, PRICE, GENERAL, FEATURES, SER&ACC`.
> NER thương hiệu/cửa hàng dùng `underthesea` (NER tổng quát PER/LOC/ORG) + gazetteer thương hiệu.

## 🏗️ Cấu trúc dự án (thực tế)
```
vietnamese-graph-rag/
├── README.md  ·  config.yaml  ·  pyproject.toml  ·  Dockerfile  ·  Makefile  ·  .env.example
├── notebooks/                                 # bản demo Kaggle (khám phá nhanh)
│   ├── kaggle_part1_embedding_ner.ipynb
│   └── kaggle_part2_graph_rag.ipynb
├── src/vngraphrag/                            # 🟢 package LLMOps (production-lite)
│   ├── config.py            # ⚙️ infra: config YAML + env (secret qua env)
│   ├── observability.py     # ⚙️ infra: log latency/token/cost + feedback → JSONL
│   ├── core/                # 📚 dữ liệu + biểu diễn
│   │   ├── data.py          #    load UIT-ViSFD + Shopee, parse nhãn, keyword/gazetteer
│   │   ├── embeddings.py    #    PhoBERT encoder (mean-pool + token-level) + maxsim
│   │   ├── index.py         #    DocumentIndex: vector + meta, lưu/nạp CÓ VERSION
│   │   └── kg.py            #    Knowledge Graph (build/save/load, query, product_context)
│   │   └── aspect_clf.py    #    🧠 model BiLSTM ĐÃ TRAIN (deploy: nạp từ artifacts/aspect_clf.pt)
│   ├── rag/                 # 🔗 suy luận
│   │   ├── retrieval.py     #    HybridRetriever (bi-encoder → MaxSim → graph boost)
│   │   ├── generate.py      #    OpenAI client + prompt grounding
│   │   └── pipeline.py      #    GraphRAGPipeline: orchestrate + observability + classify_aspects
│   └── cli/                 # ⌨️ entrypoints
│       ├── build_index.py   #    build & persist index + KG
│       ├── train_aspect.py  #    🧠 TRAIN BiLSTM → artifacts/aspect_clf.pt (không cần notebook)
│       ├── evaluate.py      #    P@k/MRR ablation + F1 + REGRESSION GATE cho CI
│       └── import_artifacts.py  # xác nhận artifacts xuất từ notebook
├── app/
│   ├── api.py           # FastAPI: /health /query /feedback
│   └── ui.py            # Gradio UI (in-process hoặc gọi API) + nút 👍/👎
├── tests/test_core.py   # unit test no-GPU (chạy trong CI)
├── .github/workflows/ci.yml   # lint + test; eval-gate chạy thủ công
├── data/raw/            # UIT-ViSFD + shopee_reviews_full.csv
├── artifacts/           # (sinh ra) index + KG + metrics.json — gitignored
├── logs/                # (sinh ra) queries.jsonl + feedback.jsonl — gitignored
└── report/
```

## 🔧 Pipeline LLMOps (production-lite)

> 📐 Kiến trúc chi tiết (sơ đồ module + data-flow + data contracts + design decisions): xem [ARCHITECTURE.md](ARCHITECTURE.md).

```
            build_index (CLI)                  serve
data/raw ───────────────────► artifacts/ ──► GraphRAGPipeline ──► FastAPI /query ──► Gradio UI
  UIT-ViSFD + Shopee          (index+KG       retrieve → graph        │  observability      │ 👍/👎
                               versioned)     → OpenAI generate       └─ logs/queries.jsonl ─┘ feedback.jsonl
                                                     ▲
                              evaluate (CLI) ── P@k/MRR + regression gate (CI)
```

| Thành phần LLMOps | Hiện thực |
|---|---|
| Config + secret tách env | `config.py` + `.env` (key chỉ qua `OPENAI_API_KEY`) |
| Artifact versioning | `index.py` manifest (hash model + #records) → tự rebuild khi lệch |
| **Train → deploy model** | BiLSTM train ở notebook §6 → lưu `artifacts/aspect_clf.pt` → `core/aspect_clf.py` nạp → phục vụ qua `POST /classify` (fallback keyword nếu chưa có model) |
| Serving API | `app/api.py` (FastAPI: `/query` `/classify` `/feedback` `/health`) |
| UI + feedback loop | `app/ui.py` (Gradio, 👍/👎 → `logs/feedback.jsonl`) |
| Observability | mỗi query log latency + token + **cost USD** vào `logs/queries.jsonl` |
| Eval + regression gate | `evaluate.py` → `metrics.json`, exit≠0 nếu MRR < `eval_min_mrr` |
| Đóng gói + CI | `Dockerfile` + `.github/workflows/ci.yml` |

### Chạy local (không phải Kaggle)
```bash
pip install -e ".[dev]"          # cài package
make check                        # 👀 1 lệnh kiểm soát: syntax mọi .py + JSON notebook + inventory
export OPENAI_API_KEY=sk-...      # (Windows PowerShell: $env:OPENAI_API_KEY="sk-...")

# Đường A — tính tại chỗ (cần GPU cho nhanh):
make train                        # train BiLSTM → artifacts/aspect_clf.pt (deploy /classify + làm giàu KG)
make index                        # encode ~10.9k review → artifacts/ (lần sau nạp lại, không tính lại)
# Đường B — tái dùng kết quả notebook Kaggle:
#   chạy notebook Part 2 §7 → tải artifacts/ về repo →
make import                       # xác nhận artifacts/ hợp lệ (khỏi encode lại)

make eval                         # P@k/MRR + regression gate → artifacts/metrics.json
make api                          # FastAPI http://localhost:8000  (POST /query)
make ui                           # Gradio UI (link demo)
make test                         # unit test no-GPU
docker build -t vngraphrag .      # đóng gói serving
```
**Kiểm soát code:** `make check` in ra toàn bộ file + số dòng + vai trò và báo lỗi syntax — chạy bất cứ lúc nào để khỏi phải tự dò file.

## 🚀 Chạy trên Kaggle
1. Tạo Kaggle Notebook mới, upload `notebooks/kaggle_part1_embedding_ner.ipynb`.
2. **Settings**: bật **Internet ON** và **Accelerator = GPU T4**.
3. (Part 2) Thêm OpenAI API key: **Add-ons → Secrets → New secret**, tên `OPENAI_API_KEY`
   (lấy tại https://platform.openai.com/api-keys). Mặc định model `gpt-4o-mini` — đổi `LLM_MODEL` trong notebook nếu muốn `gpt-4o`/`gpt-3.5-turbo`. Không có key vẫn chạy được phần Retrieval + Graph.
4. Run All. Part 1 → `data/train_processed.csv` + `results_part1.json`; Part 2 tự chứa (không cần output Part 1).

Chạy local:
```bash
pip install -r requirements.txt
```

## 🔧 Các lỗi đã sửa so với bản đầu
| # | Lỗi | Cách sửa |
|---|---|---|
| 1 | Regex `\{(\w+)#(\w+)\}` làm **mất aspect SER&ACC** (1.995 lần) | Đổi thành `\{([\w&]+)#(\w+)\}` |
| 2 | Attention reranker `nn.Linear` **chưa train** → nhiễu ngẫu nhiên | Thay bằng **late-interaction MaxSim** (ColBERT-style, không cần train) |
| 3 | **Rò rỉ metric**: boost theo đúng nhãn đang chấm điểm | Boost theo aspect **quan sát trong nội dung**; chấm điểm bằng **nhãn vàng** (độc lập) |
| 4 | Aspect query chỉ khớp tiếng Anh | Thêm **bảng ánh xạ keyword tiếng Việt → aspect** |
| 5 | So sánh embedding bằng "avg cosine" (vô nghĩa) | Thay bằng **Precision@k & MRR** |
| 6 | NER tuyên bố sai domain | Dùng `underthesea` + gazetteer, **nối brand vào KG**, mô tả trung thực |
| 7 | README/báo cáo lệch thực tế | Đồng bộ cấu trúc + điền số liệu từ notebook |

## 📄 License
Academic use only — NLP Final Project 2025-2026.
