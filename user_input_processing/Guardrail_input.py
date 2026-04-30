import json
import ollama

INPUT_GUARDRAIL_PROMPT = """Bạn là Bộ lọc Đầu vào của hệ thống tư vấn pháp luật Việt Nam.
Dữ liệu của hệ thống bao gồm 11 văn bản luật:
1. Hiến pháp 2013 (quyền công dân, tổ chức nhà nước, đảng phái)
2. Bộ luật Dân sự 2015 (hợp đồng, tài sản, thừa kế, bồi thường)
3. Luật An ninh mạng 2018 (nội dung mạng, tội phạm mạng)
4. Luật Giao thông đường bộ 2008 (phương tiện, vi phạm giao thông)
5. Luật Căn cước công dân 2014 (CCCD, giấy tờ tùy thân)
6. Luật Thuế thu nhập cá nhân (thuế TNCN, giảm trừ gia cảnh)
7. Luật Thuế thu nhập doanh nghiệp (thuế TNDN, ưu đãi thuế)
8. Luật Bảo hiểm xã hội 2014 (BHXH, BHYT, thai sản, hưu trí)
9. Bộ luật Lao động 2019 (lương, thưởng, hợp đồng lao động, sa thải, nghỉ phép, lương tháng 13)
10. Luật Bảo vệ quyền lợi người tiêu dùng (khiếu nại, đổi trả, bảo hành)
11. Luật Thương mại điện tử (mua bán online, hóa đơn điện tử)

NHIỆM VỤ:
1. Kiểm tra ngôn ngữ: Chỉ chấp nhận tiếng Việt.
2. Kiểm tra phạm vi: Chấp nhận nếu câu hỏi liên quan đến BẤT KỲ nội dung nào trong 11 luật trên.
   - CHẤP NHẬN: "lương tháng 13" (Lao động), "quyền tự do ngôn luận" (Hiến pháp), "hợp đồng mua bán" (Dân sự), "lập đảng" (Hiến pháp)
   - TỪ CHỐI: câu hỏi về y tế, giáo dục, thể thao, nấu ăn không liên quan pháp lý
3. KHI NGHI NGỜ, hãy CHẤP NHẬN (is_in_scope=true).
4. clean_query: sao chép NGUYÊN VĂN câu hỏi đầu vào, chỉ bỏ phần rõ ràng ngoài luật.

ĐỊNH DẠNG TRẢ VỀ (CHỈ JSON):
{
  "is_vietnamese": true/false,
  "is_in_scope": true/false,
  "detected_domains": ["tên các luật liên quan"],
  "clean_query": "Câu hỏi gốc hoặc phần hợp lệ — KHÔNG được viết mô tả hay nhận xét",
  "rejection_reason": "Lý do từ chối nếu có (tiếng Việt)"
}"""

# Keywords that strongly indicate one of the 11 supported law domains.
# Used as safety net when the LLM incorrectly rejects in-scope input.
_LEGAL_KEYWORDS = [
    'luật', 'điều', 'khoản', 'nghị định', 'thông tư', 'hiến pháp',
    'dân sự', 'hình sự', 'lao động', 'lương', 'thuế', 'bảo hiểm',
    'hợp đồng', 'tòa án', 'tố tụng', 'quyền', 'nghĩa vụ', 'vi phạm',
    'người tiêu dùng', 'thương mại', 'an ninh mạng', 'giao thông',
    'căn cước', 'công dân', 'doanh nghiệp', 'sa thải', 'nghỉ phép',
    'thưởng', 'tháng 13', 'bhxh', 'bhyt', 'hưu trí', 'thai sản',
    'đảng', 'bầu cử', 'biểu tình', 'ngôn luận', 'báo chí',
]

_META_PATTERNS = [
    "phần câu hỏi",
    "thuộc phạm vi",
    "ngoài phạm vi",
    "không thuộc",
    "đây là",
    "câu hỏi này",
    "văn bản này",
    "đoạn văn",
]


def _has_legal_keywords(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _LEGAL_KEYWORDS)


def _validate_clean_query(clean_query: str | None, original: str) -> str:
    """
    Guard against the LLM writing a meta-comment in clean_query instead of
    copying/cleaning the actual input text. Falls back to original if so.
    """
    if not clean_query or not clean_query.strip():
        return original

    lower = clean_query.lower()
    if any(p in lower for p in _META_PATTERNS):
        return original

    # If the "cleaned" result is suspiciously short relative to original, keep original.
    if len(original) > 30 and len(clean_query.strip()) < len(original) * 0.25:
        return original

    return clean_query.strip()


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

        if not result.get("is_vietnamese"):
            return {"status": "REJECT", "message": "Hệ thống chỉ hỗ trợ tiếng Việt."}

        if not result.get("is_in_scope"):
            # Secondary keyword safety net — override incorrect LLM rejection.
            if _has_legal_keywords(user_input):
                clean_q = _validate_clean_query(result.get("clean_query"), user_input)
                return {"status": "PASS", "clean_query": clean_q}
            return {
                "status":  "REJECT",
                "message": "Câu hỏi nằm ngoài phạm vi 11 văn bản luật hỗ trợ.",
            }

        clean_q = _validate_clean_query(result.get("clean_query"), user_input)
        return {"status": "PASS", "clean_query": clean_q}

    except Exception:
        return {"status": "PASS", "clean_query": user_input}