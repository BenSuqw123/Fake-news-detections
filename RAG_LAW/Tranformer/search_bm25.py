import pickle
import numpy as np
from pathlib import Path
from underthesea import word_tokenize

BM25_DATA = None

def load_bm25():
    global BM25_DATA

    if BM25_DATA is None:
        path = Path(__file__).resolve().parent.parent / "Models" / "bm25" / "bm25_database.pkl"
        with open(path, "rb") as f:
            BM25_DATA = pickle.load(f)

    return BM25_DATA


def search_bm25(query: str, top_k=3):
    try:
        data = load_bm25()
        bm25 = data["bm25_model"]
        corpus = data["metadata"]
        tokenized_query = word_tokenize(query, format="text").lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue

            doc = corpus[idx]

            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": {
                    "law_title": doc.get("law_title"),
                    "article": doc.get("article"),
                    "url": doc.get("url")
                },
                "score": float(scores[idx]),
                "method": "bm25"
            })

        return results

    except Exception as e:
        print(f"BM25 error: {e}")
        return []