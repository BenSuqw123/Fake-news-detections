import json
import ollama
INPUT_GUARDRAIL_PROMPT = """Bạn là Bộ lọc Đầu vào của hệ thống tư vấn pháp luật Việt Nam.
Dữ liệu của hệ thống CHỈ bao gồm 11 văn bản luật: Hiến pháp, Dân sự, An ninh mạng, Giao thông, Căn cước, Thuế TNCN, Thuế TNDN, BHXH, Lao động, BV quyền lợi người tiêu dùng, TMĐT.

NHIỆM VỤ:
1. Kiểm tra ngôn ngữ: Chỉ chấp nhận tiếng Việt.
2. Kiểm tra phạm vi: Chỉ chấp nhận các câu hỏi liên quan đến 11 luật trên.
3. Phân tách ý định: Nếu câu hỏi có phần thuộc luật và phần không, phải tách rõ.

ĐỊNH DẠNG TRẢ VỀ (CHỈ JSON):
{
  "is_vietnamese": true/false,
  "is_in_scope": true/false,
  "detected_domains": ["tên các luật liên quan"],
  "clean_query": "Phần câu hỏi thuộc phạm vi luật (loại bỏ phần linh tinh)",
  "rejection_reason": "Lý do từ chối nếu có (tiếng Việt)"
}"""

async def check_input_validity(user_input: str) -> dict:
    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': INPUT_GUARDRAIL_PROMPT},
                {'role': 'user', 'content': user_input}
            ],
            options={'temperature': 0},
            format='json'
        )
        
        result = json.loads(response['message']['content'])
        
        # Logic xử lý kết quả
        if not result.get("is_vietnamese"):
            return {"status": "REJECT", "message": "Hệ thống chỉ hỗ trợ tiếng Việt."}
            
        if not result.get("is_in_scope"):
            return {"status": "REJECT", "message": "Câu hỏi nằm ngoài phạm vi 11 văn bản luật hỗ trợ."}
            
        return {"status": "PASS", "clean_query": result.get("clean_query")}
        
    except Exception as e:
        # Fallback nếu LLM lỗi
        return {"status": "PASS", "clean_query": user_input}