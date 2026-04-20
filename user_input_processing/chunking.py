import json
import ollama
import asyncio

# --- HÀM CHUNKING ONLINE (MỚI) ---
def split_text_into_chunks(text, max_chars=1000):
    """
    Chia nhỏ văn bản đầu vào nếu nó quá dài trước khi gửi cho AI.
    """
    chunks = []
    # Tách theo các dấu hiệu pháp lý hoặc xuống dòng
    raw_chunks = text.split("\n") 
    current_chunk = ""
    
    for segment in raw_chunks:
        if len(current_chunk) + len(segment) < max_chars:
            current_chunk += segment + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = segment + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- PROMPT ĐIỀU KHIỂN ---
SYSTEM_PROMPT = """You are a strict information extraction engine for a Vietnamese Legal Fact-Checking system.
Your task is to convert Vietnamese text into atomic, verifiable claims. STRICT JSON ONLY."""

async def process_chunk(client, chunk_text):
    """Gửi từng mảnh nhỏ cho AI xử lý"""
    try:
        response = await client.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Trích xuất TOÀN BỘ các ý định pháp lý: {chunk_text}"}
            ],
            options={'temperature': 0},
            format='json'
        )
        data = json.loads(response['message']['content'].strip())
        return data.get("claims", [])
    except:
        return []

async def extract_atomic_claims(article_text: str) -> list:
    if not article_text or len(article_text.strip()) < 10:
        return []

    client = ollama.AsyncClient()
    
    # 1. BƯỚC CHUNKING ONLINE: Chia nhỏ đầu vào
    text_chunks = split_text_into_chunks(article_text, max_chars=1200)
    
    # 2. XỬ LÝ SONG SONG CÁC MẢNH (Nếu văn bản quá dài)
    tasks = [process_chunk(client, chunk) for chunk in text_chunks]
    all_raw_results = await asyncio.gather(*tasks)

    # 3. HẬU XỬ LÝ VÀ GOM NHÓM
    final_claims = []
    for claims_list in all_raw_results:
        for c in claims_list:
            if c.get("claim") and len(c["claim"]) > 5:
                clean_text = c["claim"].strip()
                if not clean_text.endswith(('.', '?', '!')):
                    clean_text += '.'
                
                final_claims.append({
                    "claim": clean_text,
                    "entities": c.get("entities", []),
                    "type": c.get("type", "FACT"),
                    "is_verifiable": c.get("is_verifiable", True)
                })
    
    return final_claims