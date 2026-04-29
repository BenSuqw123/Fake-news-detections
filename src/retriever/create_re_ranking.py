from __future__ import annotations
from typing import List, Sequence
import numpy as np


def sigmoid_normalize(score: float) -> float:
    return float(1.0 / (1.0 + np.exp(-score)))


def _bm25_normalize(score: float) -> float:
    """Map raw BM25 score (0–∞) to 0–1. Typical range 0–15; score=5 → ~0.73."""
    return sigmoid_normalize(0.5 * (score - 3.0))


def rerank_documents(query: str, docs: Sequence[dict], top_k: int | None = None) -> List[dict]:
    """
    Score-based pseudo-rerank: no CrossEncoder (avoids 2.3 GB OOM on limited RAM).
    Uses rrf_score > normalized_score > BM25 raw score, all mapped to 0–1.
    """
    if not docs:
        return []

    reranked: List[dict] = []
    for doc in docs:
        item = dict(doc)
        if "rrf_score" in item:
            # RRF scores are small (0.01–0.03); scale to 0–1 with sigmoid
            item["rerank_score"] = sigmoid_normalize(item["rrf_score"] * 200 - 3.0)
        elif "normalized_score" in item:
            item["rerank_score"] = float(item["normalized_score"])
        else:
            # Raw BM25 score (5–15 typical range)
            item["rerank_score"] = _bm25_normalize(float(item.get("score", 0.0)))
        item["rerank_raw_score"] = float(item.get("score", 0.0))
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k] if top_k is not None else reranked


# load_reranker kept so startup pre-warm in main.py doesn't crash
def load_reranker():
    return None
