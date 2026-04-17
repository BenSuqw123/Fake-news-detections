from pathlib import Path
import sys
import ollama
import numpy as np
from underthesea import word_tokenize 
sys.path.insert(0, r"D:\Fake-news-detections")
from RAG_LAW.Tranformer.search_chormadb import search_chroma
from RAG_LAW.Tranformer.search_bm25 import search_bm25


def validate_and_route_input(user_question: str) -> str:
    system_prompt = """You are a strict classifier. 
Analyze the input and output EXACTLY one of these four terms: 
- LAW (for legal, traffic rules, court, crimes)
- HISTORY (for past events, wars, dynasties, historical figures)
- NOT_ENGLISH (if the text is not in English)
- NOT_INCLUDE (if it is about anything else like cooking, tech, general chat)
Rules:
- Output ONLY the uppercase word.
- No punctuation, no explanation."""

    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ])
        raw_result = response['message']['content'].strip().upper().replace(".", "")
        result = raw_result.split()[0] if raw_result else "NOT_INCLUDE"
        
        return result if result in ["LAW", "HISTORY", "NOT_ENGLISH"] else "NOT_INCLUDE"
    except Exception:
        return "NOT_INCLUDE"

def extract_key_info(user_input: str) -> str:
    prompt = """Task: Extract search keywords for a semantic database.
Rules:
1. Extract core entities, main actions, and specific identifiers.
2. Maintain technical or specialized terminology found in the input.
3. Output ONLY keywords separated by commas."""
    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': user_input}
    ])
    keys = response['message']['content'].strip()
    return keys


def filter_relevant_docs(query, docs):
    keywords = set(word_tokenize(query.lower(), format="text").split())
    filtered = []
    for doc in docs:
        text = doc["text"].lower()
        overlap = sum(1 for kw in keywords if kw in text)
        if overlap >= 3:
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
        evidence_ratio = overlap_count / query_len
        
        if evidence_ratio > best_evidence:
            best_evidence = evidence_ratio

        if doc["method"] == "vector":
            sim = 1.0 - doc["score"]
            if sim > best_vector:
                best_vector = sim
                
        elif doc["method"] == "bm25":
            norm_bm25 = min(doc["score"] / 20.0, 1.0) 
            if norm_bm25 > best_bm25:
                best_bm25 = norm_bm25

    final_score = (0.3 * best_vector) + (0.3 * best_bm25) + (0.4 * best_evidence)

    if final_score <= 0:
        return 0.0

    return round(final_score * 100, 2)

def process_fake_news_query(user_input: str):
    print(f"\nInput: {user_input}")
    clean_query = extract_key_info(user_input)
    print(f"Extracted: {clean_query}")
    final_query = f"{user_input}, {clean_query}"
    print(f"Final Query: {final_query}")
    docs = hybrid_search(final_query, top_k=5)
    if not docs:
        print("Không tìm được tài liệu")
        return []

    docs = filter_relevant_docs(user_input, docs)
    print(f"Retrieved (filtered): {len(docs)} docs")
    return docs

def hybrid_search(query, top_k=5):
    vector_docs = search_chroma(query, top_k)
    bm25_docs = search_bm25(query, top_k)
    all_docs = vector_docs + bm25_docs
    seen = set()
    final_docs = []
    for doc in all_docs:
        doc_id = doc["id"]
        if doc_id not in seen:
            final_docs.append(doc)
            seen.add(doc_id)
        if len(final_docs) >= top_k:
            break
    return final_docs

def verify_claim(user_input, docs):
    if not docs:
        return {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "confidence": "0%",
            "evidence": [],
            "reason": "Không tìm thấy tài liệu"
        }

    confidence = calculate_confidence(docs, user_input)
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
Bạn là hệ thống kiểm chứng pháp luật, yêu cầu suy luận CHÍNH XÁC và TUÂN THỦ LOGIC.

======================
TUYÊN BỐ:
"{user_input}"
======================

TÀI LIỆU:
{context}

======================
QUY TẮC BẮT BUỘC:

1. Nếu có ít nhất một điều luật KHẲNG ĐỊNH trực tiếp nội dung tuyên bố
→ VERDICT PHẢI là TRUE

2. Nếu có điều luật PHỦ ĐỊNH trực tiếp nội dung tuyên bố
→ VERDICT PHẢI là FALSE

3. Nếu KHÔNG có điều luật nào liên quan trực tiếp
→ VERDICT = NOT_ENOUGH_EVIDENCE

======================
NGHIÊM CẤM:

- Không được trả NOT_ENOUGH_EVIDENCE nếu đã có điều luật khẳng định rõ
- Không được suy diễn ngoài nội dung tài liệu
- Không được dùng điều luật không liên quan để phản bác
- Không được tạo mâu thuẫn giữa các điều luật
- Không được bỏ qua điều luật quan trọng

======================
CÁCH LÀM:

- Chỉ chọn điều luật liên quan trực tiếp đến tuyên bố
- Ưu tiên điều luật chứa các từ khóa giống tuyên bố
- Nếu một điều luật đã đủ để kết luận → không cần xét thêm

======================
OUTPUT (CHỈ 3 DÒNG):

VERDICT: TRUE hoặc FALSE hoặc NOT_ENOUGH_EVIDENCE
EVIDENCE: Điều + Văn bản (ví dụ: Điều 1 Hiến pháp năm 2013)
REASON: Giải thích ngắn gọn dựa trực tiếp vào evidence

======================
LƯU Ý QUAN TRỌNG:

Nếu evidence chứa nội dung khẳng định rõ (ví dụ: "độc lập", "chủ quyền")
→ bắt buộc VERDICT = TRUE
"""

    try:
        response = ollama.chat(
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
    
if __name__ == "__main__":
    query = "Việt Nam là một quốc gia độc lập và có chủ quyền"
    docs = process_fake_news_query(query)

    if docs:
        result = verify_claim(query, docs)
        print("\nRESULT:\n")
        print(f"Verdict     : {result['verdict']}")
        print(f"Confidence  : {result['confidence']}")
        print(f"Evidence    : {result['evidence']}")
        print(f"Reason      : {result['reason']}")
       