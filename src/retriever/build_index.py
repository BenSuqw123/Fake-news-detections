import sys
from pathlib import Path
import chromadb
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODELS_DIR
from src.retriever.embedder import BGEEmbedder

CHROMA_CLIENTS = {}
embedder = None 

def validate_and_route_input(user_question: str) -> str:
    system_prompt = """You are a strict input validator and router for a specific database.
Your task is to evaluate the user's input based on two strict criteria: Language and Topic.

CRITERIA & RULES:
1. LANGUAGE CHECK: If the input is NOT written in English, you MUST respond with: "NOT_ENGLISH"
2. TOPIC CHECK: If the input IS in English, determine its topic:
   - If it is about laws, legal regulations, court cases, or legal procedures -> respond with: "LAW"
   - If it is about historical events, past eras, or historical figures -> respond with: "HISTORY"
   - If it is about ANYTHING ELSE -> respond with: "NOT_INCLUDE"

OUTPUT FORMAT:
- Respond with exactly ONE word from this list: [NOT_ENGLISH, LAW, HISTORY, NOT_INCLUDE].
- Do not explain, do not add punctuation."""

    try:
        response = ollama.chat(model='llama3.2', messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ])
        
        result = response['message']['content'].strip().upper()
        
        if result in ["LAW", "HISTORY", "NOT_ENGLISH", "NOT_INCLUDE"]:
            return result
        else:
            return "NOT_INCLUDE"
            
    except Exception:
        return "NOT_INCLUDE"

def extract_key_info(user_input: str) -> str:
    prompt = """Extract the core entities and main claim from the text.
Output ONLY a comma-separated list of the most important keywords in English. No full sentences."""

    response = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': user_input}
    ])
    return response['message']['content'].strip()

def search_chroma(query_vector: list, domain: str, top_k: int = 3) -> list:
    global CHROMA_CLIENTS
    
    domain = domain.lower()
    db_path = str(MODELS_DIR / domain)
    
    if domain not in CHROMA_CLIENTS:
        CHROMA_CLIENTS[domain] = chromadb.PersistentClient(path=db_path)

    client = CHROMA_CLIENTS[domain]
    
    try:
        collection = client.get_collection(name=domain)
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    retrieved_docs = []
    if results and 'documents' in results and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            doc_text = results['documents'][0][i]
            doc_meta = results['metadatas'][0][i] if results['metadatas'] else {}
            
            retrieved_docs.append({
                'text': doc_text,
                'source': doc_meta.get('source', domain),
                'title': doc_meta.get('title', ''),
                'url': doc_meta.get('url', '')
            })
            
    return retrieved_docs

def verify_fake_news(user_input: str, retrieved_docs: list) -> str:
    context_text = "\n\n".join([f"Source ({doc.get('source')}): {doc.get('text')}" for doc in retrieved_docs])
    
    prompt = f"""You are a Fact-Checker.
Here is some official information extracted from the database:
--- OFFICIAL DOCUMENTS ---
{context_text}
-----------------------------

Based EXACTLY on the documents above, verify if the following user input is TRUE or FALSE:
Input to verify: "{user_input}"

Requirements:
1. Conclude immediately: [TRUE] or [FALSE] or [NOT ENOUGH INFORMATION].
2. Briefly explain based on the reference documents above."""

    response = ollama.chat(model='llama3.2', messages=[
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

    if embedder is None: 
        embedder = BGEEmbedder()
    
    query_vector = embedder.embed_documents([clean_query])[0]
    
    docs = search_chroma(query_vector, domain, top_k=3)
    
    if not docs:
        return "No matching data found in the database."
        
    final_result = verify_fake_news(user_input, docs)
    return final_result

if __name__ == "__main__":
    query = "What is the penalty for running a red light in California?"
    result = process_fake_news_query(query)
    print(result)