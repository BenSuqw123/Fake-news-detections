import json
import ollama
import asyncio

# --- PROMPT ĐIỀU KHIỂN CHẶT CHẼ ---
SYSTEM_PROMPT = """You are a strict information extraction engine for a Vietnamese Legal Fact-Checking system.
Your task is to convert Vietnamese text into atomic, verifiable claims.

CORE RULES:
1. ATOMIC: Each claim must contain EXACTLY ONE fact.
2. STANDALONE: Replace pronouns (nó, điều này, họ) with specific entities (Hiến pháp, Chính phủ).
3. FACT-ONLY: Extract only statements about laws, history, or statistics. Skip opinions.
4. LANGUAGE: Keep claims in Vietnamese.

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "claims": [
    {
      "claim": "Chuỗi nội dung claim (Tiếng Việt)",
      "entities": ["thực thể 1", "thực thể 2"],
      "type": "FACT | LEGAL | STATISTIC",
      "is_verifiable": true
    }
  ]
}"""

async def extract_atomic_claims(article_text: str) -> list:
    """
    Trích xuất các tuyên bố đơn lẻ (atomic claims) từ văn bản tiếng Việt.
    """
    if not article_text or len(article_text.strip()) < 10:
        return []

    client = ollama.AsyncClient()
    
    try:
        response = await client.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f"Trích xuất TOÀN BỘ các ý định pháp lý từ văn bản sau, không bỏ sót bất kỳ vế nào: {article_text}"}
            ],
            options={
                'temperature': 0,
                'num_predict': 2048, 
                'top_p': 0.1
            },
            format='json'
        )
        
        content = response['message']['content'].strip()
        
        # Parse JSON an toàn
        data = json.loads(content)
        raw_claims = data.get("claims", [])
        
        # --- HẬU XỬ LÝ (Post-processing) ---
        final_claims = []
        for c in raw_claims:
            # Chỉ lấy các claim có đủ thông tin và có khả năng kiểm chứng
            if c.get("claim") and len(c["claim"]) > 5:
                # Đảm bảo claim kết thúc bằng dấu chấm nếu thiếu
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

    except json.JSONDecodeError:
        print("❌ Lỗi: AI không trả về định dạng JSON chuẩn.")
        return []
    except Exception as e:
        print(f"❌ Lỗi trích xuất claim: {str(e)}")
        return []


if __name__ == "__main__":
    async def test():
        text = "Hiến pháp 2013 nói Việt Nam độc lập và Chính phủ có 15 bộ ngành."
        print(f"Input: {text}")
        results = await extract_atomic_claims(text)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    asyncio.run(test())