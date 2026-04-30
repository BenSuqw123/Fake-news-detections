"""
search_bm25.py
==============
BM25 retriever with thread-safe singleton and dual-retrieval support.

search_bm25()               — supporting evidence
search_bm25_contradiction() — contradicting evidence (same search, LLM judges)

Tokenizer: hybrid (underthesea compounds + clean syllables) — must match
create_bm25.py exactly so index tokens and query tokens are in the same space.
"""
import re
import pickle
import threading
import numpy as np
from pathlib import Path
from underthesea import word_tokenize
from src.config import TOP_K_RETRIEVAL

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BM25_PATH    = _PROJECT_ROOT / "RAG_LAW" / "Models" / "bm25" / "bm25_database.pkl"

BM25_DATA  = None
_BM25_LOCK = threading.Lock()

# Boost recall for known tricky compound terms.
# Hybrid tokenizer handles most cases — this map only adds high-IDF
# compound tokens that are unlikely to surface from syllable matching alone.
_EXPANSION_MAP = {
    'đảng_phái':     ['đảng', 'đảng_cộng_sản_việt_nam', 'tiên_phong', 'lực_lượng', 'lãnh_đạo'],
    'chính_trị':     ['hiến_pháp', 'lãnh_đạo'],
    'tổng_thống':    ['chủ_tịch_nước'],
    'bầu_cử':        ['ứng_cử', 'bầu'],
    'biểu_tình':     ['tụ_tập', 'hội_họp'],
    'tự_do_báo_chí': ['báo_chí', 'ngôn_luận'],
}


def _tokenize_query(text: str) -> list[str]:
    """
    Hybrid tokenizer — mirrors create_bm25.tokenize() exactly.
    Must stay in sync with the index tokenizer or BM25 scores are meaningless.
    """
    compound_tokens = word_tokenize(text, format="text").lower().split()
    clean_text      = re.sub(r"[^\w\s]", " ", text.lower())
    syllable_tokens = clean_text.split()

    seen   = set()
    result = []
    for t in compound_tokens + syllable_tokens:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def load_bm25():
    global BM25_DATA

    if BM25_DATA is not None:
        return BM25_DATA

    with _BM25_LOCK:
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
    """Shared helper: tokenize query and return top_k scored doc dicts."""
    data   = load_bm25()
    bm25   = data["bm25_model"]
    corpus = data["metadata"]

    tokenized_query = _tokenize_query(query_text)

    # Query expansion — boost score for known tricky compound terms
    expanded = tokenized_query.copy()
    for token in tokenized_query:
        if token in _EXPANSION_MAP:
            for extra in _EXPANSION_MAP[token]:
                if extra not in expanded:
                    expanded.append(extra)

    scores  = bm25.get_scores(expanded)
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
    """Search for SUPPORTING evidence."""
    try:
        return _bm25_search_raw(query, top_k, method_tag="bm25")
    except Exception as e:
        print(f"BM25 error: {e}")
        return []


def search_bm25_contradiction(query: str, top_k: int = 5) -> list:
    """
    Search for CONTRADICTING evidence.
    Uses the same query — LLM judges what contradicts from the retrieved docs.
    No negation keyword injection (unreliable, pollutes BM25 scores).
    """
    try:
        return _bm25_search_raw(query, top_k, method_tag="bm25_contra")
    except Exception as e:
        print(f"BM25 contradiction error: {e}")
        return []