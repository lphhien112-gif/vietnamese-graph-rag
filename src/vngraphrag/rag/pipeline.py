"""GraphRAGPipeline: orchestrates retrieve -> graph context -> generate, with observability.

Loads persisted index + KG (rebuilds if missing/stale). This is the single object the
API and UI depend on.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..core import (
    DocumentIndex,
    aspect_from_query,
    build_kg,
    build_records,
    confidence_note,
    detect_brand,
    graph_query,
    graph_query_brand,
    load_kg,
    load_shopee,
    load_visfd,
    product_context,
    save_kg,
)
from ..observability import QueryLogger, Timer, estimate_cost
from .generate import Generator, build_prompt
from .retrieval import HybridRetriever


class GraphRAGPipeline:
    def __init__(self, cfg: Config, encoder, index: DocumentIndex, kg, generator: Generator, aspect_clf=None):
        self.cfg = cfg
        self.encoder = encoder
        self.index = index
        self.kg = kg
        self.retriever = HybridRetriever(index, encoder, cfg)
        self.generator = generator
        self.aspect_clf = aspect_clf  # model BiLSTM đã train (None nếu chưa deploy)
        self.logger = QueryLogger(cfg.logs_dir)

    # ---------------------------------------------------------------
    @classmethod
    def from_artifacts(cls, cfg: Config | None = None, rebuild: bool = False) -> GraphRAGPipeline:
        cfg = cfg or Config.load()
        from ..core import PhoBERTEncoder
        from ..core.aspect_clf import AspectClassifier

        encoder = PhoBERTEncoder(cfg.embedding_model, cfg.max_seq_len)
        aspect_clf = AspectClassifier.load(cfg.artifacts_dir, cfg.aspect_clf_threshold)

        index = None if rebuild else DocumentIndex.load(cfg.artifacts_dir, cfg.embedding_model)
        kg_path = Path(cfg.artifacts_dir) / "kg.pkl"
        if index is None or not kg_path.exists():
            visfd, shopee = load_visfd(cfg.data_dir), load_shopee(cfg.data_dir)
            if index is None:
                records = build_records(visfd, shopee)
                index = DocumentIndex.build(records, encoder, cfg.embedding_model)
                index.save(cfg.artifacts_dir)
            kg = build_kg(visfd, shopee, aspect_clf)  # model deploy vào KG
            save_kg(kg, kg_path)
        else:
            kg = load_kg(kg_path)

        return cls(cfg, encoder, index, kg, Generator(cfg), aspect_clf)

    # ---------------------------------------------------------------
    def classify_aspects(self, text: str) -> list[str]:
        """Dùng model BiLSTM đã deploy để dự đoán aspect (fallback: keyword)."""
        if self.aspect_clf is not None:
            return sorted(self.aspect_clf.predict([text])[0])
        from ..core import aspects_from_text

        return sorted(aspects_from_text(text))

    # ---------------------------------------------------------------
    def _graph_context(self, question: str) -> str:
        asp = aspect_from_query(question)
        gctx = ""
        if asp:
            # Nếu câu hỏi nhắc 1 hãng -> thống kê RIÊNG hãng đó; nếu hãng không có dữ
            # liệu về aspect này thì quay về thống kê toàn corpus (tránh trả rỗng).
            brand = detect_brand(question)
            sd, label = None, asp
            if brand != "Unknown":
                bsd = graph_query_brand(self.kg, brand, asp)
                if bsd:
                    sd, label = bsd, f"{asp} ({brand})"
            if sd is None:
                sd = graph_query(self.kg, asp)
            tot = sum(v["count"] for v in sd.values())
            for _node, inf in sorted(sd.items(), key=lambda x: -x[1]["count"]):
                pct = 100 * inf["count"] / tot if tot else 0
                gctx += f"- {label}/{inf['sentiment']}: {inf['count']} review ({pct:.0f}%)\n"
            # Cảnh báo độ tin cậy: mẫu quá nhỏ (n<30) hoặc aspect dữ liệu mỏng (STORAGE)
            note = confidence_note(tot, asp)
            if note:
                gctx += f"  (lưu ý: {note})\n"
        for prod, avg, ncnt in product_context(self.kg, question):
            gctx += f"- Sản phẩm '{prod}': {avg:.1f} sao ({ncnt} review)\n"
        return gctx

    def answer(self, question: str, top_k: int | None = None) -> dict:
        with Timer() as t:
            retrieved = self.retriever.retrieve(question, top_k=top_k)
            gctx = self._graph_context(question)
            prompt = build_prompt(question, retrieved, gctx)
            gen = self.generator.generate(prompt)
        cost = estimate_cost(self.cfg.llm.model, gen["prompt_tokens"], gen["completion_tokens"], self.cfg.llm.prices)
        qid = self.logger.log_query(
            {
                "question": question,
                "model": self.cfg.llm.model,
                "latency_ms": t.ms,
                "n_retrieved": len(retrieved),
                "prompt_tokens": gen["prompt_tokens"],
                "completion_tokens": gen["completion_tokens"],
                "cost_usd": cost,
                "sources": [r["source"] for r in retrieved],
            }
        )
        return {
            "id": qid,
            "question": question,
            "answer": gen["text"],
            "graph_context": gctx,
            "retrieved": retrieved,
            "latency_ms": t.ms,
            "cost_usd": cost,
        }

    def feedback(self, query_id: str, rating: int, note: str = ""):
        self.logger.log_feedback(query_id, rating, note)
