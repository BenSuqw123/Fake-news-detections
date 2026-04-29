import json
import ollama
import asyncio

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.

Nhiệm vụ: Phân tích đoạn văn đầu vào và trích xuất các tuyên bố có thể kiểm chứng được về mặt pháp lý.

Quy tắc trích xuất:
1. Mỗi tuyên bố phải là một luận điểm độc lập, có thể kiểm tra riêng lẻ
2. Giữ nguyên các số điều khoản (Điều X, Khoản Y), tên luật, tên cơ quan
3. Loại bỏ ý kiến cá nhân, cảm xúc, không có cơ sở pháp lý
4. Nếu đoạn văn đề cập nhiều điều luật khác nhau, tách thành nhiều claims riêng
5. Keywords phải bao gồm số điều khoản nếu có

Trả về JSON với định dạng sau:
{
  "claims": [
    {
      "claim": "<tuyên bố nguyên văn, đủ ngữ cảnh>",
      "keywords": "<các từ khóa pháp lý quan trọng, cách nhau bởi dấu phẩy. Ưu tiên: số điều khoản (Điều X), tên luật, tên quyền/nghĩa vụ cụ thể>",
      "entities": ["<thực thể pháp lý 1>", "<thực thể pháp lý 2>"],
      "type": "constitutional|civil|criminal|labor|administrative",
      "is_verifiable": true
    }
  ]
}

Ví dụ keywords tốt:
- Input: "Theo Điều 25 Hiến pháp 2013, công dân có quyền tự do ngôn luận"
  keywords: "Điều 25, Hiến pháp 2013, quyền tự do ngôn luận"
- Input: "lập đảng phái chính trị độc lập"
  keywords: "đảng phái chính trị, lập đảng, Điều 4, độc lập"

Chỉ trả về JSON, không thêm bất kỳ text nào khác."""


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

        data = json.loads(content)
        raw_claims = data.get("claims", [])

        final_claims = []
        for c in raw_claims:
            if c.get("claim") and len(c["claim"]) > 5:
                clean_text = c["claim"].strip()
                if not clean_text.endswith(('.', '?', '!')):
                    clean_text += '.'

                final_claims.append({
                    "claim":         clean_text,
                    "keywords":      c.get("keywords", ""),
                    "entities":      c.get("entities", []),
                    "type":          c.get("type", "FACT"),
                    "is_verifiable": c.get("is_verifiable", True),
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
