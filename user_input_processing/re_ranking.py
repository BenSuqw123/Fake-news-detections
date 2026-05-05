import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever.create_re_ranking import rerank_documents

def apply_reranking(query, docs, top_k=None):
    if not docs:
        return []

    try:
        return rerank_documents(query, docs, top_k=top_k)
    except Exception as e:
        print(f"Re-ranking error: {e}")
        fallback_docs = [dict(doc) for doc in docs]
        for doc in fallback_docs:
            doc["rerank_score"] = doc.get("rrf_score", doc.get("normalized_score", 0.0))
            doc["rerank_raw_score"] = doc.get("score", 0.0)

        fallback_docs.sort(
            key=lambda item: (
                item.get("rrf_score", 0.0),
                item.get("normalized_score", 0.0),
            ),
            reverse=True,
        )

        if top_k is not None:
            return fallback_docs[:top_k]
        return fallback_docs

def reciprocal_rank_fusion(result_lists, top_k=50, k=60):
    fused_scores = {}
    fused_docs = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

            if doc_id not in fused_docs:
                fused_doc = dict(doc)
                fused_doc["sources"] = [doc.get("method", "unknown")]
                fused_docs[doc_id] = fused_doc
            else:
                existing_doc = fused_docs[doc_id]
                method = doc.get("method", "unknown")
                if method not in existing_doc["sources"]:
                    existing_doc["sources"].append(method)

                if doc.get("normalized_score", 0.0) > existing_doc.get("normalized_score", 0.0):
                    existing_doc["text"] = doc["text"]
                    existing_doc["metadata"] = doc["metadata"]
                    existing_doc["score"] = doc["score"]
                    existing_doc["method"] = method
                    existing_doc["normalized_score"] = doc.get("normalized_score", 0.0)

    final_docs = []
    for doc_id, doc in fused_docs.items():
        fused_doc = dict(doc)
        fused_doc["rrf_score"] = fused_scores.get(doc_id, 0.0)
        final_docs.append(fused_doc)

    final_docs.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return final_docs[:top_k]
