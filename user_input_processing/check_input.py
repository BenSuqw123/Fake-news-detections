import sys
import chromadb
import ollama

sys.path.insert(0, r"D:\Fake-news-detections")
from src.retriever.embedder import BGEEmbedder

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

import re

def format_search_payload(raw_keywords: str, topic: str) -> dict:
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
        "vector_query": ordered_keys, 
        "metadata_filter": {          
            "topic": topic.lower().strip()
        }
    }

def search_chroma(payload: dict, top_k: int = 3) -> list:
    global CHROMA_CLIENT
    global embedder
    db_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\VectorBD\chroma_db"
    
    if CHROMA_CLIENT is None:
        CHROMA_CLIENT = chromadb.PersistentClient(path=db_path)
    if embedder is None: 
        embedder = BGEEmbedder()

    try:
        collection = CHROMA_CLIENT.get_collection(name="fake_news_rag")
        query_vector = embedder.embed_documents([payload['vector_query']])[0]

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=payload['metadata_filter']  
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
        print(f"Lỗi truy vấn ChromaDB: {e}")
        return []
    
def verify_fake_news(user_input: str, retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "Verdict: NOT ENOUGH INFORMATION\nReasoning: Không tìm thấy tài liệu trong Database."

    context_parts = []
    for doc in retrieved_docs:
        content = doc.get('text', 'No content')
        meta = doc.get('metadata', {})
        source_title = meta.get('title', 'Unknown Source')
        
        context_parts.append(f"SOURCE [{source_title}]: {content}")

    context_text = "\n\n".join(context_parts)
    
    prompt = f"""You are a Fact-Checking Bot. Verify the Claim using ONLY the Source Data.

--- SOURCE DATA ---
{context_text}
-------------------

Claim: "{user_input}"

INSTRUCTIONS:
1. Extract ALL sentences from the Source Data that mention keywords in the Claim.
2. Compare the technical definitions and historical facts in the Source against the Claim.
3. If the Source is truncated (ends abruptly), only verify based on the available text.

OUTPUT FORMAT:
- Evidence Found: (Paste exact sentence)
- Verdict: [TRUE / FALSE / NOT ENOUGH INFORMATION]
- Reasoning: (Short explanation)"""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': 'You are a strict fact-checker. Use only provided text.'},
        {'role': 'user', 'content': prompt}
    ])
    
    return response['message']['content'].strip()


def process_fake_news_query(user_input: str) -> str:
    global embedder
    
    status = validate_and_route_input(user_input)
    
    if status == "NOT_ENGLISH":
        return "Error: Please enter your query in English only."
    elif status == "NOT_INCLUDE":
        return "Error: Input does not belong to the Law or History database."

    domain = status.lower()
    clean_query = extract_key_info(user_input)
    payload=format_search_payload(clean_query, domain)
    print(payload)
    docs = search_chroma(payload, top_k=5)
    return verify_fake_news(payload, docs)

if __name__ == "__main__":
    query = " Military history is the study of armed conflict in the history of humanity, and its impact on the societies, cultures and economies thereof, as well as the resulting changes to local and international relationships. Professional historians normally focus on military affairs that had a major impact on the societies involved as well as the aftermath of conflicts, while amateur historians and hobbyists often take a larger interest in the details of battles, equipment, and uniforms in use. The essen"
    # kq = validate_and_route_input(query)
    # print(kq)
    # key=extract_key_info(query)
    # print(key)
    # sx = format_search_payload(key, kq)
    # print(sx)
    # domain=kq.lower
    test = process_fake_news_query(query)
    print(test)