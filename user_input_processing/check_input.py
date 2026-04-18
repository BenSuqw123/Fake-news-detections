from pathlib import Path
import sys
import ollama
import asyncio
from underthesea import word_tokenize
import re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_LAW.Tranformer.search_chormadb import search_chroma
from RAG_LAW.Tranformer.search_bm25 import search_bm25
from src.retriever.create_re_ranking import rerank_documents
from user_input_processing.claim_extractor import extract_atomic_claims

HYBRID_CANDIDATE_K = 50
RERANK_TOP_K = 5


async def async_extract_key_info(user_input: str) -> str:
    prompt = """Task: Extract search keywords for a semantic database.
Rules:
1. Extract core entities, main actions, and specific identifiers.
2. Maintain technical or specialized terminology found in the input.
3. Output ONLY keywords separated by commas."""
    client = ollama.AsyncClient()
    response = await client.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    )
    keys = response["message"]["content"].strip()
    return keys


def normalize_bm25_query(query: str) -> str:
    tokens = word_tokenize(query.lower(), format="text").split()

    legal_noise = {
        "là", "của", "và", "có", "được", "theo", "về", "cho", "với", "trong",
        "một", "các", "những", "này", "đó", "khi", "thì", "tại", "do", "ở",
        "quy_định", "pháp_luật", "điều", "khoản", "chương", "nghị_định", "thông_tư",
        "căn_cứ", "việc", "cho_biết", "tình_huống", "hỏi"
    }

    filtered_tokens = [t for t in tokens if t not in legal_noise and len(t) > 1]

    if not filtered_tokens:
        return query

    return " ".join(filtered_tokens)



def calculate_confidence(docs, query):
    if not docs:
        return 0.0

    query_tokens = set(word_tokenize(query.lower(), format="text").split())
    query_len = len(query_tokens)

    if query_len == 0:
        return 0.0

    best_rerank = 0.0
    best_evidence = 0.0

    for doc in docs:
        text_tokens = set(word_tokenize(doc["text"].lower(), format="text").split())
        overlap_count = len(query_tokens.intersection(text_tokens))
        evidence_ratio = overlap_count / query_len if query_len > 0 else 0

        if evidence_ratio > best_evidence:
            best_evidence = evidence_ratio

        rerank_score = doc.get("rerank_score")
        if rerank_score is None:
            rerank_score = doc.get("normalized_score", 0.0)
        if rerank_score > best_rerank:
            best_rerank = rerank_score

    final_score = (0.8 * best_rerank) + (0.2 * best_evidence)

    if final_score <= 0:
        return 0.0

    return round(max(0.0, min(final_score * 100, 100.0)), 2)


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
                fused_doc["sources"] = [doc["method"]]
                fused_docs[doc_id] = fused_doc
            else:
                existing_doc = fused_docs[doc_id]
                if doc["method"] not in existing_doc["sources"]:
                    existing_doc["sources"].append(doc["method"])

                if doc.get("normalized_score", 0.0) > existing_doc.get("normalized_score", 0.0):
                    existing_doc["text"] = doc["text"]
                    existing_doc["metadata"] = doc["metadata"]
                    existing_doc["score"] = doc["score"]
                    existing_doc["method"] = doc["method"]
                    existing_doc["normalized_score"] = doc.get("normalized_score", 0.0)

    final_docs = []
    for doc_id, doc in fused_docs.items():
        fused_doc = dict(doc)
        fused_doc["rrf_score"] = fused_scores.get(doc_id, 0.0)
        final_docs.append(fused_doc)

    final_docs.sort(key=lambda item: item.get("rrf_score", 0.0), reverse=True)
    return final_docs[:top_k]


def hybrid_rrf_search(vector_query, bm25_query=None, top_k=RERANK_TOP_K, candidate_k=HYBRID_CANDIDATE_K):
    if bm25_query is None:
        bm25_query = vector_query

    vector_docs = search_chroma(vector_query, candidate_k)
    bm25_docs = search_bm25(bm25_query, candidate_k)

    for doc in vector_docs:
        doc["normalized_score"] = max(1.0 - doc["score"], 0.0)

    for doc in bm25_docs:
        doc["normalized_score"] = min(doc["score"] / 20.0, 1.0)

    fused_docs = reciprocal_rank_fusion(
        [vector_docs, bm25_docs],
        top_k=candidate_k,
    )
    reranked_docs = apply_reranking(vector_query, fused_docs, top_k=top_k)
    return reranked_docs[:top_k]


async def async_process_fake_news_query(user_input: str):
    clean_query = await async_extract_key_info(user_input)
    vector_query = f"{user_input}, {clean_query}"
    bm25_query = normalize_bm25_query(user_input)

    docs = await asyncio.to_thread(
        hybrid_rrf_search,
        vector_query,
        bm25_query,
        RERANK_TOP_K,
        HYBRID_CANDIDATE_K,
    )
    if not docs:
        return []

    return docs[:RERANK_TOP_K]

async def async_verify_claim(user_input, docs):
    if not docs:
        return {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "confidence": "0%",
            "evidence": "",
            "reason": "Không tìm thấy tài liệu liên quan.",
        }

    confidence = calculate_confidence(docs, user_input)
    if confidence < 20:
        return {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "confidence": f"{confidence}%",
            "evidence": "",
            "reason": "Mức độ tin cậy của tài liệu quá thấp để đưa ra kết luận bảo đảm.",
        }

    context = "\n\n".join(
        [
            f"""
[QUAN TRỌNG]
- Văn bản: {doc['metadata'].get('law_title', '')}
- Điều: {doc['metadata'].get('article', '')}
- Nội dung: {doc['text']}
"""
            for doc in docs
        ]
    )

    prompt = f"""
Bạn là hệ thống kiểm chứng thông tin pháp luật. Yêu cầu kiểm chứng khách quan, dựa TRỰC TIẾP và DUY NHẤT vào TÀI LIỆU được cung cấp.

======================
TUYÊN BỐ:
"{user_input}"
======================

TÀI LIỆU:
{context}

======================
QUY TẮC CỐT LÕI (BẮT BUỘC):
1. VỀ NGUỒN GỐC: Chỉ dựa vào tài liệu được cung cấp. Tuyệt đối KHÔNG suy diễn bằng kiến thức bên ngoài.
2. BẰNG CHỨNG XÁC THỰC (TRUE): Để kết luận TRUE, tài liệu phải khẳng định rõ ràng, trọn vẹn ngữ nghĩa của tuyên bố. Sự trùng khớp của một vài từ khóa đơn lẻ là KHÔNG ĐỦ.
3. BẰNG CHỨNG PHẢN BÁC (FALSE): Để kết luận FALSE, tài liệu phải chứa nội dung mâu thuẫn trực tiếp hoặc phủ định thẳng sự việc trong tuyên bố.
4. THIẾU BẰNG CHỨNG (NOT_ENOUGH_EVIDENCE): Nếu tài liệu chỉ đề cập gián tiếp, một phần hoặc hoàn toàn thiếu cơ sở để kết luận chắc chắn sự việc -> BẮT BUỘC trả về NOT_ENOUGH_EVIDENCE.

======================
OUTPUT (CHỈ TRẢ VỀ 3 DÒNG THEO ĐÚNG FORMAT SAU):

VERDICT: [Điền TRUE hoặc FALSE hoặc NOT_ENOUGH_EVIDENCE]
EVIDENCE: [Liệt kê 1-3 căn cứ pháp lý quan trọng nhất, ngăn cách bằng dấu chấm phẩy. Nếu chọn NOT_ENOUGH_EVIDENCE thì để N/A]
REASON: [Tóm tắt tổng hợp từ toàn bộ tài liệu đã cho, nêu căn cứ mạnh nhất, vai trò của các tài liệu còn lại.]
"""

    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )

        output = response["message"]["content"]
        verdict = "NOT_ENOUGH_EVIDENCE"
        evidence = ""
        reason = ""

        for line in output.split("\n"):
            line = line.strip()

            if line.startswith("VERDICT"):
                verdict = line.split(":", 1)[-1].strip()
            elif line.startswith("EVIDENCE"):
                evidence = line.split(":", 1)[-1].strip()
            elif line.startswith("REASON"):
                reason = line.split(":", 1)[-1].strip()

        return {
            "verdict": verdict,
            "confidence": f"{confidence}%",
            "evidence": evidence,
            "reason": reason,
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence": "0%",
            "evidence": "",
            "reason": str(e),
        }


async def process_single_claim(claim_data: dict, semaphore: asyncio.Semaphore):
    async with semaphore:
        claim_text = claim_data.get("claim", "")
        ctype = claim_data.get("type", "UNKNOWN")
        print(f'\n--- Processing Claim: "{claim_text}" [{ctype}] ---')

        docs = await async_process_fake_news_query(claim_text)
        result = await async_verify_claim(claim_text, docs)

        result["claim"] = claim_text
        result["type"] = ctype
        return result


async def process_article(article_text: str):
    print("Extracting atomic claims from the text...")
    claims = await extract_atomic_claims(article_text)

    if not claims:
        print("No valid claims could be extracted.")
        return

    verifiable_claims = [c for c in claims if c.get("is_verifiable")]
    print(f"\n[EXTRACTED] {len(claims)} claims found. {len(verifiable_claims)} are verifiable.")

    semaphore = asyncio.Semaphore(3)
    tasks = [process_single_claim(claim, semaphore) for claim in verifiable_claims]

    results = await asyncio.gather(*tasks)

    print("\n\n=== SUMMARY REPORT ===")
    for i, res in enumerate(results, 1):
        print(f"Claim {i}: {res['claim']}")
        print(f"Type: {res['type']}")
        print(f"Verdict: {res['verdict']} | Confidence: {res['confidence']}")
        print(f"Evidence: {res['evidence']}")
        print(f"Reason: {res['reason']}\n")


if __name__ == "__main__":
    sample_text = "Hiến pháp năm 2013 quy định Việt Nam là một quốc gia độc lập và có chủ quyền. Đây là quy định có tính lịch sử."
    asyncio.run(process_article(sample_text))
