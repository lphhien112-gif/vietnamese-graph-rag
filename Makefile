.PHONY: install check index train eval import api ui test lint format docker

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

eval:             ## run eval harness + regression gate
	python -m vngraphrag.cli.evaluate

api:              ## serve FastAPI
	uvicorn app.api:app --host 0.0.0.0 --port 8000

ui:               ## launch Gradio UI (in-process)
	python -m app.ui

test:
	pytest -q tests/test_core.py

lint:
	ruff check src app tests scripts

format:           ## auto-format toàn bộ code theo ruff
	ruff format src app tests scripts
	ruff check --fix src app tests scripts

docker:
	docker build -t vngraphrag .
