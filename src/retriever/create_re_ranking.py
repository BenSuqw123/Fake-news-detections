from __future__ import annotations
from typing import List, Sequence
import numpy as np

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return np.asarray(scores, dtype=float).tolist()


_RERANKER: BGEReranker | None = None


def load_reranker() -> BGEReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = BGEReranker()
    return _RERANKER


def sigmoid_normalize(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-score)))


def rerank_documents(query: str, docs: Sequence[dict], top_k: int | None = None) -> List[dict]:
    if not docs:
        return []

    reranker = load_reranker()
    texts = [doc.get("text", "") for doc in docs]
    raw_scores = reranker.score(query, texts)

    reranked_docs: List[dict] = []
    for doc, raw_score in zip(docs, raw_scores):
        item = dict(doc)
        item["rerank_raw_score"] = float(raw_score)
        item["rerank_score"] = sigmoid_normalize(float(raw_score))
        reranked_docs.append(item)

    reranked_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    if top_k is not None:
        return reranked_docs[:top_k]
    return reranked_docs
