import json
from pathlib import Path
import sys
import chromadb
import ollama
import pickle
from pyvi import ViTokenizer
import numpy as np
import re

sys.path.insert(0, r"D:\Fake-news-detections")
from src.retriever.embedder import BGEM3Embedder

CHROMA_CLIENT = None
embedder = None 

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


def format_search_payload(raw_keywords: str) -> dict:
    key_list = [k.strip() for k in raw_keywords.split(",")]
    time_pattern = r"(\d{4}|century|thế kỷ|th\d+|\b[IVXLCDM]+\b|^\d{1,2}(st|nd|rd|th)$)"
    
    time_keys = []
    content_keys = []
    for key in key_list:
        if re.search(time_pattern, key, re.IGNORECASE):
            time_keys.append(key)
        else:
            content_keys.append(key)
            
    ordered_keys = ", ".join(list(dict.fromkeys(time_keys + content_keys)))
    
    return {
        "vector_query": ordered_keys
    }

def search_chroma(payload: dict, top_k: int = 3) -> list:
    global CHROMA_CLIENT, embedder
    db_path = r"D:\Fake-news-detections\RAG-LAW\Models\law_chroma"
    
    if CHROMA_CLIENT is None:
        CHROMA_CLIENT = chromadb.PersistentClient(path=db_path)
    if embedder is None: 
        from src.retriever.embedder import BGEM3Embedder
        embedder = BGEM3Embedder(device='cpu')

    try:
        collection = CHROMA_CLIENT.get_collection(name="law")
        query_vector = embedder.embed_documents([payload['vector_query']])[0]

        results = collection.query(
            query_embeddings=[query_vector.tolist()], 
            n_results=top_k
        )
        
        retrieved_docs = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                retrieved_docs.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        return retrieved_docs
    except Exception as e:
        print(f"Lỗi ChromaDB: {e}")
        return []
    
    
def search_bm25(raw_keywords: str, top_k: int = 3) -> list:
    try:
        DATA_PATH = Path(r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_cleaned.json")
        BM25_PATH = Path(r"D:\Fake-news-detections\RAG-LAW\Models\bm25\bm25_model.pkl")
        with open(BM25_PATH, 'rb') as f:
            bm25 = pickle.load(f)
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            articles = json.load(f)
        search_terms = [k.strip().lower() for k in raw_keywords.split(",")]
        
        processed_query = []
        for term in search_terms:
            processed_query.extend(ViTokenizer.tokenize(term).split())

        scores = bm25.get_scores(processed_query)
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        bm25_docs = []
        for idx in top_n_indices:
            if scores[idx] > 0: 
                bm25_docs.append({
                    'text': articles[idx].get('content', ''),
                    'metadata': articles[idx],
                    'score': scores[idx],
                    'method': 'BM25'
                })
        return bm25_docs
    except Exception as e:
        print(f"Lỗi BM25: {e}")
        return []

def process_fake_news_query(user_input: str):
    clean_query = extract_key_info(user_input)
    payload = format_search_payload(clean_query)
    
    vector_docs = search_chroma(payload, top_k=3)
    bm25_docs = search_bm25(clean_query, top_k=3)
    
    combined = []
    for v, b in zip(vector_docs, bm25_docs):
        combined.append(v)
        combined.append(b)
    combined.extend(vector_docs[len(bm25_docs):])
    combined.extend(bm25_docs[len(vector_docs):])

    unique_content = []
    seen_texts = set()

    for doc in combined:
        content = doc.get('text', '').strip()
        normalized_content = " ".join(content.lower().split())
        
        if normalized_content and normalized_content not in seen_texts:
            unique_content.append(content)
            seen_texts.add(normalized_content)
        
        if len(unique_content) >= 3:
            break
            
    return unique_content

def verify_claim(user_input, retrieved_docs):
    if not retrieved_docs:
        return "Kết luận: NOT ENOUGH EVIDENCE (Không tìm thấy tài liệu liên quan trong hệ thống)."

    context = "\n\n".join([f"--- TÀI LIỆU {i+1} ---\n{text}" for i, text in enumerate(retrieved_docs)])
    
    prompt = f"""
Bạn là một Thẩm phán chuyên về kiểm chứng thông tin pháp luật.
Nhiệm vụ: Dựa trên các tài liệu được cung cấp, hãy đưa ra phán quyết về tính xác thực của TUYÊN BỐ.

--- TUYÊN BỐ (CLAIM) ---
"{user_input}"

--- TÀI LIỆU CĂN CỨ ---
{context}

--- YÊU CẦU ĐẦU RA ---
1. **KẾT LUẬN**: [TRUE / FALSE / NOT ENOUGH EVIDENCE]
2. **LÝ LUẬN PHÁP LÝ**: Viết một đoạn văn ngắn (không liệt kê) tổng hợp các căn cứ từ tài liệu để giải thích tại sao tuyên bố đó Đúng hoặc Sai. Nếu có sự mâu thuẫn giữa các tài liệu, hãy ưu tiên tài liệu có nội dung cụ thể nhất về vấn đề được hỏi.
3. **TRÍCH DẪN NGUỒN**: Chỉ liệt kê các Điều/Khoản trực tiếp làm căn cứ cho kết luận trên.

--- TRẢ LỜI ---
"""

    try:
        response = ollama.chat(
            model='llama3.2', 
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0,      
                'top_p': 0.1,         
                'num_ctx': 4096        
            }
        )
        return response['message']['content']
    except Exception as e:
        return f"Lỗi hệ thống: {e}"
    
if __name__ == "__main__":
    # query = "Theo quy định tại văn bản hợp nhất 33/VBHN-VPQH 2026, Nhà nước bảo đảm kinh phí cho hoạt động quy hoạch đô thị và nông thôn dựa trên các quy định của pháp luật về ngân sách nhà nước."
    query='Bộ trưởng Bộ Xây dựng là người có thẩm quyền trực tiếp phê duyệt dự toán kinh phí hoạt động quy hoạch đô thị và nông thôn cho tất cả các đơn vị trực thuộc trên toàn quốc.'
    # 1. Tìm tài liệu (Hybrid Search)
    docs = process_fake_news_query(query)
    
    if isinstance(docs, list):
        # 2. Đánh giá đúng/sai
        result = verify_claim(query, docs)
        print(result)