"""
search_chromadb.py
==================
ChromaDB retriever with thread-safe singleton and dual-retrieval support.

search_chroma()              — supporting evidence (existing)
search_chroma_contradiction() — contradicting evidence (new)
"""
import chromadb
import threading
from pathlib import Path
from src.config import TOP_K_RETRIEVAL

# Resolve ChromaDB path relative to project root (works on any machine)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _PROJECT_ROOT / "RAG_LAW" / "Models" / "law_chroma"

CHROMA_CLIENT = None
_CHROMA_LOCK = threading.Lock()  # prevents race-condition double-init


def load_chroma():
    global CHROMA_CLIENT

    # Fast path: already initialised (no lock needed — reads are safe)
    if CHROMA_CLIENT is not None:
        from src.retriever.embedder import get_embedder
        return CHROMA_CLIENT.get_collection("law"), get_embedder()

    with _CHROMA_LOCK:
        # Double-checked locking: another thread may have initialised while
        # we waited for the lock.
        if CHROMA_CLIENT is None:
            if not _DB_PATH.exists():
                raise FileNotFoundError(
                    f"[ChromaDB] Database directory not found: {_DB_PATH}\n"
                    "Make sure RAG_LAW/Models/law_chroma/ exists in the project root."
                )
            # Use forward slashes: chromadb's Rust/SQLite layer on Windows
            # can misparse backslashes in long paths with spaces.
            db_path_str = _DB_PATH.as_posix()
            CHROMA_CLIENT = chromadb.PersistentClient(path=db_path_str)

    from src.retriever.embedder import get_embedder
    collection = CHROMA_CLIENT.get_collection("law")
    return collection, get_embedder()


def _query_chroma(query_text: str, top_k: int) -> list:
    """
    Shared helper: embed query_text and return top_k docs as dict list.
    score = cosine similarity in [0, 1] (higher = more similar).
    """
    collection, embedder = load_chroma()

    query_vec = embedder.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
    )

    docs = []
    for i in range(len(results["documents"][0])):
        distance   = results["distances"][0][i]
        similarity = 1.0 - (distance / 2.0)   # cosine distance → similarity [0, 1]
        docs.append({
            "id":       results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    similarity,
            "method":   "vector",
        })

    # Sort by similarity descending
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs


def search_chroma(query: str, top_k: int = TOP_K_RETRIEVAL) -> list:
    """Search for SUPPORTING evidence — standard semantic search."""
    try:
        return _query_chroma(query, top_k)
    except Exception as e:
        print(f"Chroma error: {e}")
        return []


def search_chroma_contradiction(query: str, top_k: int = 5) -> list:
    """
    Search for CONTRADICTING evidence.

    Strategy: prefix the query with negation language so the embedding
    gravitates toward docs that discuss what is NOT allowed / NOT mandated.
    This is heuristic — the negation prefix shifts the embedding toward
    the "prohibition/restriction" semantic neighbourhood in the law corpus.
    """
    negation_query = (
        f"Quy định KHÔNG cho phép hoặc KHÔNG công nhận, bị CẤM, "
        f"không có căn cứ pháp lý: {query}"
    )
    try:
        docs = _query_chroma(negation_query, top_k)
        # Mark as contradiction evidence so downstream code can distinguish
        for d in docs:
            d["method"] = "vector_contra"
        return docs
    except Exception as e:
        print(f"Chroma contradiction error: {e}")
        return []