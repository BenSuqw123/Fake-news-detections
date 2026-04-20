from user_input_processing.search_chormadb import search_chroma
from user_input_processing.search_bm25 import search_bm25
from user_input_processing.re_ranking import apply_reranking

HYBRID_CANDIDATE_K = 50
RERANK_TOP_K = 5

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

def hybrid_rrf_search(vector_query, bm25_query=None, top_k=5):
    # Khởi tạo danh sách rỗng để tránh NoneType lỗi sau này
    vector_docs, bm25_docs = [], []
    
    try:
        vector_docs = search_chroma(vector_query, 20) or []
    except Exception as e:
        print(f"❌ Lỗi Chroma: {e}")

    try:
        bm25_docs = search_bm25(bm25_query or vector_query, 20) or []
    except Exception as e:
        print(f"❌ Lỗi BM25: {e}")

    # Nếu cả hai đều lỗi, trả về danh sách trống thay vì để Pipeline chạy tiếp
    if not vector_docs and not bm25_docs:
        return []

    # Tiếp tục RRF và Reranking...
    fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
    return apply_reranking(vector_query, fused, top_k=top_k)