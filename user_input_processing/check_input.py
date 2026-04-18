from pathlib import Path
import sys
import ollama
import numpy as np
import asyncio
from underthesea import word_tokenize 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RAG_LAW.Tranformer.search_chormadb import search_chroma
from RAG_LAW.Tranformer.search_bm25 import search_bm25
from user_input_processing.claim_extractor import extract_atomic_claims

async def async_extract_key_info(user_input: str) -> str:
    prompt = """Task: Extract search keywords for a semantic database.
Rules:
1. Extract core entities, main actions, and specific identifiers.
2. Maintain technical or specialized terminology found in the input.
3. Output ONLY keywords separated by commas."""
    client = ollama.AsyncClient()
    response = await client.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': user_input}
    ])
    keys = response['message']['content'].strip()
    return keys

def filter_relevant_docs(query, docs):
    keywords = set(word_tokenize(query.lower(), format="text").split())
    filtered = []
    query_len = len(keywords)
    for doc in docs:
        text = doc["text"].lower()
        overlap = sum(1 for kw in keywords if kw in text)
        overlap_ratio = overlap / query_len if query_len > 0 else 0
        if overlap >= 1 or overlap_ratio > 0.2:
            filtered.append(doc)

    return filtered

def calculate_confidence(docs, query):
    if not docs:
        return 0.0

    query_tokens = set(word_tokenize(query.lower(), format="text").split())
    query_len = len(query_tokens)
    
    if query_len == 0:
        return 0.0

    best_vector = 0.0
    best_bm25 = 0.0
    best_evidence = 0.0

    for doc in docs:
        text_tokens = set(doc["text"].lower().split())
        overlap_count = len(query_tokens.intersection(text_tokens))
        evidence_ratio = overlap_count / query_len if query_len > 0 else 0
        
        if evidence_ratio > best_evidence:
            best_evidence = evidence_ratio

        if doc["method"] == "vector":
            sim = doc.get("normalized_score", max(1.0 - doc["score"], 0.0))
            if sim > best_vector:
                best_vector = sim
                
        elif doc["method"] == "bm25":
            norm_bm25 = doc.get("normalized_score", min(doc["score"] / 20.0, 1.0))
            if norm_bm25 > best_bm25:
                best_bm25 = norm_bm25

    # Refactored exact weighting formula
    final_score = (0.5 * best_vector) + (0.3 * best_bm25) + (0.2 * best_evidence)

    if final_score <= 0:
        return 0.0

    return round(max(0.0, min(final_score * 100, 100.0)), 2)

def hybrid_search(query, top_k=5):
    vector_docs = search_chroma(query, top_k)
    bm25_docs = search_bm25(query, top_k)
    
    # Pre-normalize for sorting
    for doc in vector_docs:
        doc["normalized_score"] = max(1.0 - doc["score"], 0.0)
    for doc in bm25_docs:
        doc["normalized_score"] = min(doc["score"] / 20.0, 1.0)
        
    all_docs = vector_docs + bm25_docs
    seen = set()
    final_docs = []
    
    for doc in all_docs:
        doc_id = doc["id"]
        if doc_id not in seen:
            final_docs.append(doc)
            seen.add(doc_id)
            
    # Deduplicate and sort descending by normalized score
    final_docs.sort(key=lambda x: x.get("normalized_score", 0.0), reverse=True)
    return final_docs[:top_k]

async def async_process_fake_news_query(user_input: str):
    clean_query = await async_extract_key_info(user_input)
    final_query = f"{user_input}, {clean_query}"
    
    # Offload blocking searches to thread pool
    docs = await asyncio.to_thread(hybrid_search, final_query, 5)
    if not docs:
        return []

    docs = filter_relevant_docs(user_input, docs)
    return docs

async def async_verify_claim(user_input, docs):
    if not docs:
        return {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "confidence": "0%",
            "evidence": "",
            "reason": "Không tìm thấy tài liệu liên quan."
        }

    # Early rejection threshold
    confidence = calculate_confidence(docs, user_input)
    if confidence < 20:
        return {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "confidence": f"{confidence}%",
            "evidence": "",
            "reason": "Mức độ tin cậy của tài liệu quá thấp để đưa ra kết luận bảo đảm."
        }

    context = "\n\n".join([
    f"""
[QUAN TRỌNG]
- Văn bản: {doc['metadata'].get('law_title', '')}
- Điều: {doc['metadata'].get('article', '')}
- Nội dung: {doc['text']}
"""
    for doc in docs
])
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
4. THIẾU BẰNG CHỨNG (NOT_ENOUGH_EVIDENCE): Nếu tài liệu chỉ đề cập gián tiếp, một phần hoặc hoàn toàn thiếu cơ sở để kết luận chắc chắn sự việc → BẮT BUỘC trả về NOT_ENOUGH_EVIDENCE.

======================
OUTPUT (CHỈ TRẢ VỀ 3 DÒNG THEO ĐÚNG FORMAT SAU):

VERDICT: [Điền TRUE hoặc FALSE hoặc NOT_ENOUGH_EVIDENCE]
EVIDENCE: [Trích dẫn Điều + Tên văn bản chứa bằng chứng. Ví dụ: Điều 1 Hiến pháp năm 2013. Nếu chọn NOT_ENOUGH_EVIDENCE thì để N/A]
REASON: [Lập luận giải thích ngắn gọn lý do tại sao theo sát luật.]
"""

    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )

        output = response['message']['content']
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
            "reason": reason
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "confidence": "0%",
            "evidence": "",
            "reason": str(e)
        }

async def process_single_claim(claim_data: dict, semaphore: asyncio.Semaphore):
    async with semaphore:
        claim_text = claim_data.get("claim", "")
        ctype = claim_data.get("type", "UNKNOWN")
        print(f"\n--- Processing Claim: \"{claim_text}\" [{ctype}] ---")
        
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
    
    # Process up to 3 claims concurrently to avoid model overload
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