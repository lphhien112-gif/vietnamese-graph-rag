.PHONY: install check index train eval eval-all import api ui ui-exp test benchmark benchmark-save lint format docker figs

install:
	pip install -e ".[dev]"

check:            ## kiểm soát toàn repo: syntax .py + JSON notebook + inventory
	python scripts/check_repo.py

import:           ## xác nhận artifacts/ (xuất từ notebook) hợp lệ để serve
	python -m vngraphrag.cli.import_artifacts

index:            ## build/persist document index + KG
	python -m vngraphrag.cli.build_index

train:            ## train BiLSTM aspect classifier -> artifacts/aspect_clf.pt
	python -m vngraphrag.cli.train_aspect

eval:             ## run eval harness + regression gate (ablation 25q + clf F1)
	python -m vngraphrag.cli.evaluate

eval-all:         ## chạy TẤT CẢ thí nghiệm cho báo cáo (cần OPENAI_API_KEY cho 2 cái cuối)
	python -m vngraphrag.cli.evaluate
	python scripts/eval_embeddings_full.py
	python scripts/eval_devtest_split.py
	python scripts/eval_rag_modes.py
	python scripts/eval_grounding.py
	python scripts/make_report_figs.py

figs:             ## sinh lại hình báo cáo từ artifacts/*.json
	python scripts/make_report_figs.py

api:              ## serve FastAPI
	uvicorn app.api:app --host 0.0.0.0 --port 8000

ui:               ## launch Gradio UI demo hỏi-đáp (in-process)
	python -m app.ui

ui-exp:           ## launch Gradio Experiment Dashboard (kết quả + so sánh 3 chế độ)
	python -m app.experiments_ui

test:             ## toàn bộ unit test no-GPU (core + data + kg + retrieval)
	pytest -q tests/test_core.py tests/test_data.py tests/test_kg.py tests/test_retrieval.py

benchmark:        ## báo cáo benchmark đầy đủ (P@k/MRR/F1 + latency + quy mô) + so baseline
	python scripts/benchmark.py

benchmark-save:   ## lưu kết quả hiện tại làm baseline mới
	python scripts/benchmark.py --save-baseline

lint:
	ruff check src app tests scripts

format:           ## auto-format toàn bộ code theo ruff
	ruff format src app tests scripts
	ruff check --fix src app tests scripts

docker:
	docker build -t vngraphrag .
