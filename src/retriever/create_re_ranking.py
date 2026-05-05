from __future__ import annotations
from typing import List, Sequence
import numpy as np


def sigmoid_normalize(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-score)))


def _bm25_normalize(score: float) -> float:
    return sigmoid_normalize(0.5 * (score - 3.0))


def rerank_documents(query: str, docs: Sequence[dict], top_k: int | None = None) -> List[dict]:
    if not docs:
        return []

    reranked: List[dict] = []
    for doc in docs:
        item = dict(doc)
        if "rrf_score" in item:
            item["rerank_score"] = sigmoid_normalize(item["rrf_score"] * 200 - 3.0)
        elif "normalized_score" in item:
            item["rerank_score"] = float(item["normalized_score"])
        else:
            item["rerank_score"] = _bm25_normalize(float(item.get("score", 0.0)))
        item["rerank_raw_score"] = float(item.get("score", 0.0))
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k] if top_k is not None else reranked


def load_reranker():
    return None
