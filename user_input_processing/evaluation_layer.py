import json
import ollama
import asyncio

# --- PROMPT ĐIỀU KHIỂN CHI TIẾT ---
EVALUATION_SYSTEM_PROMPT = """Bạn là chuyên gia thẩm định pháp lý của hệ thống Fact-checking.
Nhiệm vụ: Kiểm chứng TUYÊN BỐ dựa trên TÀI LIỆU được cung cấp.

QUY TẮC:
1. VERDICT: 
   - SUPPORTED: Tài liệu khẳng định tuyên bố.
   - CONTRADICTED: Tài liệu phủ nhận tuyên bố.
   - INSUFFICIENT: Không đủ thông tin hoặc tài liệu không liên quan.
   - PARTIAL: Đúng một phần nhưng thiếu các vế quan trọng.
2. REASON: Phải giải thích dựa trên số Điều/Khoản và tên văn bản luật.

OUTPUT JSON ONLY:
{
  "verdict": "SUPPORTED/CONTRADICTED/INSUFFICIENT/PARTIAL",
  "agreement_score": 0.0-1.0,
  "coverage_score": 0.0-1.0,
  "conflict_detected": false,
  "evidence": "Trích dẫn đoạn văn bản gốc.",
  "reason": "Giải thích logic chi tiết tại đây."
}"""

def calculate_hybrid_confidence(best_rerank, llm_output):
    """Công thức 80% Rerank + 20% LLM Logic"""
    # Tính điểm logic của LLM (0-1)
    llm_logic = float(llm_output.get("coverage_score", 0.0))
    if llm_output.get("conflict_detected", False):
        llm_logic *= float(llm_output.get("agreement_score", 0.5))
        
    # Phạt theo Verdict
    multipliers = {"SUPPORTED": 1.0, "CONTRADICTED": 1.0, "PARTIAL": 0.6, "INSUFFICIENT": 0.0}
    llm_score = llm_logic * multipliers.get(llm_output.get("verdict"), 0.0)
    
    # Tổng hợp 80/20
    final_score = (0.8 * best_rerank) + (0.2 * llm_score)
    return round(final_score * 100, 2)

async def evaluate_final(claim: str, docs: list) -> dict:
    """Hàm duy nhất dùng cho toàn bộ hệ thống"""
    # 1. Kiểm tra tài liệu thô
    if not docs:
        return _build_fallback("Không tìm thấy tài liệu liên quan trong Database.")
        
    best_rerank = max((d.get("rerank_score", d.get("normalized_score", 0.0)) for d in docs), default=0.0)
    
    # Pre-filter: Nếu điểm tìm kiếm quá thấp (< 25%), dừng luôn
    if best_rerank < 0.25:
        return _build_fallback(f"Độ liên quan của dữ liệu thô quá thấp ({round(best_rerank*100, 1)}%).")

    # 2. Format dữ liệu cho LLM
    formatted_context = "\n\n".join([
        f"[Nguồn: {d.get('metadata', {}).get('law_title')} - Điều {d.get('metadata', {}).get('article')}]: {d.get('text')}"
        for d in docs
    ])

    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': EVALUATION_SYSTEM_PROMPT},
                {'role': 'user', 'content': f"TUYÊN BỐ: {claim}\n\nTÀI LIỆU:\n{formatted_context}"}
            ],
            options={'temperature': 0},
            format='json'
        )
        
        llm_res = json.loads(response['message']['content'].strip())
        
        # 3. Tính điểm tin cậy tổng hợp
        conf_val = calculate_hybrid_confidence(best_rerank, llm_res)
        
        # 4. Đóng gói kết quả tiêu chuẩn
        result = {
            "verdict": llm_res.get("verdict", "INSUFFICIENT"),
            "confidence": f"{conf_val}%",
            "evidence": llm_res.get("evidence", "N/A"),
            "reason": llm_res.get("reason", "AI không đưa ra lý giải cụ thể."),
            "sufficient": conf_val >= 70.0 and llm_res.get("verdict") != "INSUFFICIENT"
        }
        
        # BỘ LỌC AN TOÀN: Hạ phán quyết nếu điểm thấp
        if conf_val < 70.0 and result["verdict"] == "SUPPORTED":
            result["verdict"] = "INSUFFICIENT"
            result["reason"] = f"[CẢNH BÁO ĐỘ TIN CẬY THẤP {conf_val}%]: {result['reason']}"

        return result

    except Exception as e:
        return _build_fallback(f"Lỗi phân tích: {str(e)}")

def _build_fallback(reason: str) -> dict:
    return {"verdict": "INSUFFICIENT", "confidence": "0.0%", "evidence": "N/A", "reason": reason, "sufficient": False}