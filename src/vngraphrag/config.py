"""Central configuration. Loads config.yaml then overrides with environment variables.

Secrets (OPENAI_API_KEY) are read ONLY from the environment, never from yaml.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class RetrievalConfig:
    n_candidates: int = 50
    top_k: int = 5
    w_bi: float = 0.4
    w_attn: float = 0.4
    w_graph: float = 0.2


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 400
    # USD per 1K tokens (input, output) — xấp xỉ, cập nhật theo bảng giá OpenAI
    prices: dict = field(
        default_factory=lambda: {
            "gpt-4o-mini": [0.00015, 0.0006],
            "gpt-4o": [0.0025, 0.01],
            "gpt-4.1-mini": [0.0004, 0.0016],
            "gpt-3.5-turbo": [0.0005, 0.0015],
        }
    )


@dataclass
class Config:
    data_dir: str = "data/raw"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    embedding_model: str = "vinai/phobert-base-v2"
    max_seq_len: int = 128
    aspect_clf_threshold: float = 0.5
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    # eval regression gates (CI fails if dưới ngưỡng)
    eval_min_mrr: float = 0.30
    eval_min_f1: float = 0.40

    # --- secrets via env only ---
    @property
    def openai_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | os.PathLike = "config.yaml") -> Config:
        cfg = cls()
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            cfg = _merge(cfg, raw)
        # env overrides for a few common knobs
        if os.environ.get("VNGR_LLM_MODEL"):
            cfg.llm.model = os.environ["VNGR_LLM_MODEL"]
        if os.environ.get("VNGR_ARTIFACTS_DIR"):
            cfg.artifacts_dir = os.environ["VNGR_ARTIFACTS_DIR"]
        return cfg


def _merge(cfg: Config, raw: dict) -> Config:
    for k, v in raw.items():
        if k == "retrieval" and isinstance(v, dict):
            cfg.retrieval = RetrievalConfig(**{**asdict(cfg.retrieval), **v})
        elif k == "llm" and isinstance(v, dict):
            base = asdict(cfg.llm)
            base.update(v)
            cfg.llm = LLMConfig(**base)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
