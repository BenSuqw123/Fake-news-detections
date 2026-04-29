"""
search_bm25.py
==============
BM25 retriever with thread-safe singleton and dual-retrieval support.

search_bm25()              — supporting evidence (existing)
search_bm25_contradiction() — contradicting evidence (new)
"""
import pickle
import threading
import numpy as np
from pathlib import Path
from underthesea import word_tokenize
from src.config import TOP_K_RETRIEVAL

# Resolve BM25 path relative to project root — no hardcoded Windows paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BM25_PATH = _PROJECT_ROOT / "RAG_LAW" / "Models" / "bm25" / "bm25_database.pkl"

BM25_DATA = None
_BM25_LOCK = threading.Lock()  # prevents race-condition double-load

# Prohibition / negation keywords injected for contradiction search.
# These are common Vietnamese legal phrases that appear in restrictive clauses.
_NEGATION_TERMS = [
    "không được",
    "cấm",
    "không có quyền",
    "không quy định",
    "không bắt buộc",
    "không cho phép",
    "hạn chế",
    "bị cấm",
    "không được phép",
    "vi phạm",
    "xử phạt",
]


def load_bm25():
    global BM25_DATA

    # Fast path — no lock needed for a simple read check
    if BM25_DATA is not None:
        return BM25_DATA

    with _BM25_LOCK:
        # Double-checked locking: another thread may have loaded while waiting
        if BM25_DATA is None:
            if not _BM25_PATH.exists():
                raise FileNotFoundError(
                    f"[BM25] Database not found: {_BM25_PATH}\n"
                    "Run src/retriever/create_bm25.py first."
                )
            with open(_BM25_PATH, "rb") as f:
                BM25_DATA = pickle.load(f)

    return BM25_DATA


def _bm25_search_raw(query_text: str, top_k: int, method_tag: str = "bm25") -> list:
    """Shared helper: tokenize query_text and return top_k scored doc dicts."""
    data   = load_bm25()
    bm25   = data["bm25_model"]
    corpus = data["metadata"]

    tokenized_query = word_tokenize(query_text, format="text").lower().split()
    scores  = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        doc = corpus[idx]
        results.append({
            "id":   doc["id"],
            "text": doc["text"],
            "metadata": {
                "law_title": doc.get("law_title"),
                "article":   doc.get("article"),
                "url":       doc.get("url"),
            },
            "score":  float(scores[idx]),
            "method": method_tag,
        })

    return results


def search_bm25(query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
    """Search for SUPPORTING evidence — standard BM25 keyword search."""
    try:
        return _bm25_search_raw(query, top_k, method_tag="bm25")
    except Exception as e:
        print(f"BM25 error: {e}")
        return []


def search_bm25_contradiction(query: str, top_k: int = 5) -> list:
    """
    Search for CONTRADICTING evidence.

    Strategy: append prohibition/negation keywords to the query so BM25
    scores docs that contain both query terms AND restriction vocabulary.
    This biases retrieval toward clauses that say what is NOT allowed.
    """
    negation_query = query + " " + " ".join(_NEGATION_TERMS)
    try:
        return _bm25_search_raw(negation_query, top_k, method_tag="bm25_contra")
    except Exception as e:
        print(f"BM25 contradiction error: {e}")
        return []
