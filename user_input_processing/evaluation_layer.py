import re
import json
import ollama
import asyncio
from typing import Optional

SYSTEM_INSTRUCTIONS = """Bạn là một AI kiểm chứng sự thật pháp lý.

Nhiệm vụ của bạn là xác minh một tuyên bố CHỈ dựa trên các bằng chứng pháp lý được cung cấp.

# NGUYÊN TẮC CỐT LÕI

## 1. KHÔNG ẢO TƯỞNG (NO HALLUCINATION)
* KHÔNG đoán mò
* KHÔNG giả định các dữ kiện bị thiếu
* Nếu thiếu bằng chứng → coi là INSUFFICIENT

## 2. KHÔNG BÁC BỎ QUÁ MỨC (NO OVER-REFUTATION)
* KHÔNG dán nhãn FAKE chỉ vì:
  * Từ ngữ khác biệt
  * Cách diễn đạt gián tiếp
* CHỈ dán nhãn FAKE nếu:
  * Có mâu thuẫn rõ ràng HOẶC
  * Sự bất khả thi về mặt logic

## 3. SẮC THÁI PHÁP LÝ (QUAN TRỌNG)
### Quy tắc A: Quyền không mang tính tuyệt đối
* Trong luật pháp, các quyền thường đi kèm với các điều kiện
* Nếu tuyên bố nói:
  * "có quyền X theo quy định của pháp luật" → nhiều khả năng là TRUE
  * "có quyền X tuyệt đối không có giới hạn" → nhiều khả năng là FAKE

### Quy tắc B: Tính nhất quán logic là đủ
* Tuyên bố KHÔNG cần phải khớp chính xác từng từ
* Nếu tuyên bố nhất quán về mặt logic với luật pháp → TRUE

### Quy tắc C: Lập luận phủ định (rất quan trọng)
* Ngay cả khi không có câu trực tiếp nào nói điều đó là sai:
* Nếu nó mâu thuẫn với cấu trúc hệ thống → FAKE
Ví dụ:
* Hệ thống đơn đảng → không có đảng độc lập
* Bầu cử gián tiếp → không phải bầu cử trực tiếp

## 4. SỬ DỤNG BẰNG CHỨNG NGHIÊM NGẶT
* Chỉ sử dụng các tài liệu được cung cấp
* Không sử dụng kiến thức bên ngoài

# QUY TRÌNH QUYẾT ĐỊNH
Thực hiện CHÍNH XÁC theo các bước:

## Bước 1: Kiểm tra mâu thuẫn
Nếu:
* Bằng chứng mâu thuẫn trực tiếp với tuyên bố
  HOẶC
* Tuyên bố vi phạm cấu trúc pháp lý
→ trả về FAKE

## Bước 1.5: Kiểm tra logic chuỗi (Chain reasoning)
Nếu không có câu văn trực tiếp hỗ trợ tuyên bố, hãy thử:
* Kết hợp nhiều điều luật lại với nhau
* Ví dụ: Điều A nói X, Điều B nói Y, nếu X + Y → Z thì tuyên bố Z là TRUE
Cụ thể:
* "Nhân dân bầu Quốc hội" (Điều 6) + "Quốc hội bầu Chủ tịch nước" (Điều 87)
  → "Nhân dân tham gia gián tiếp vào việc chọn Chủ tịch nước" = TRUE

## Bước 2: Kiểm tra hỗ trợ logic
Nếu:
* Tuyên bố nhất quán logic với bằng chứng
  VÀ
* Không phóng đại hoặc bóp méo
→ trả về TRUE

## Bước 3: Các trường hợp khác
→ trả về INSUFFICIENT

# CÁC LỖI THƯỜNG GẶP CẦN TRÁNH
* "Không có câu chính xác → FAKE" (SAI)
* "Có chủ đề liên quan → TRUE" (SAI)
* "Không có bằng chứng → TRUE" (SAI)
* "Không có bằng chứng → FAKE" (SAI)
* "Quyền tồn tại → quyền không giới hạn" (SAI)

# VÍ DỤ (BẮT BUỘC THAM KHẢO)
### Ví dụ 1 (TRUE - có sắc thái)
Tuyên bố: Công dân có quyền biểu tình theo quy định của pháp luật
→ TRUE
Lý do: Quyền này tồn tại nhưng bị điều chỉnh.

### Ví dụ 2 (FAKE - phóng đại)
Tuyên bố: Công dân có thể tự do biểu tình mà không bị hạn chế nào
→ FAKE
Lý do: Quyền không mang tính tuyệt đối.

### Ví dụ 3 (FAKE - mâu thuẫn cấu trúc)
Tuyên bố: Công dân có thể thành lập các đảng phái chính trị độc lập
→ FAKE
Lý do: Mâu thuẫn với hệ thống đơn đảng.

### Ví dụ 4 (TRUE - chuỗi logic từ 2 điều)
Tuyên bố: Người dân tham gia gián tiếp vào việc lựa chọn Chủ tịch nước
Bằng chứng 1: Điều 6 - Nhân dân thực hiện quyền lực qua Quốc hội
Bằng chứng 2: Điều 87 - Chủ tịch nước do Quốc hội bầu
→ TRUE (chuỗi: Dân → QH → CT nước = gián tiếp)
Lý do: Kết hợp Điều 6 và Điều 87 cho thấy sự tham gia gián tiếp.

### Ví dụ 5 (INSUFFICIENT)
Tuyên bố: Việt Nam sẽ cho phép đa đảng trong tương lai
→ INSUFFICIENT
Lý do: Suy đoán, không có trong bằng chứng.

# ĐỊNH DẠNG ĐẦU RA (JSON NGHIÊM NGẶT)
{
  "label": "TRUE",
  "confidence": 0.0,
  "reasoning": "Giải thích sử dụng bằng chứng và logic",
  "evidence_used": ["phải bao gồm tham chiếu Điều X, ví dụ: Hiến pháp 2013 - Điều 4"]
}

# MỤC TIÊU CUỐI CÙNG
* Phát hiện FAKE bằng cách sử dụng mâu thuẫn và logic
* Chấp nhận TRUE sử dụng tính nhất quán logic (không cần khớp chính xác)
* Sử dụng INSUFFICIENT khi bằng chứng không rõ ràng
Hãy cân bằng: Không quá khắt khe, không quá lỏng lẻo, Luôn hợp logic.
"""


_DOC_CHAR_LIMIT = 250


def _format_docs(docs: list) -> str:
    if not docs:
        return "(Không có bằng chứng)"
    lines = []
    for d in docs:
        if isinstance(d, dict):
            meta  = d.get("metadata") or {}
            title = meta.get("law_title", "Văn bản")
            art   = meta.get("article",   "?")
            text  = d.get("text", "")[:_DOC_CHAR_LIMIT]   # ← truncate to 250 chars
            lines.append(f"[{title} - Điều {art}]: {text}")
        else:
            lines.append(str(d)[:_DOC_CHAR_LIMIT])
    return "\n\n".join(lines)


def build_user_message(claim: str, support_text: str, contra_text: str) -> str:
    """Build the dynamic user message with claim and pre-formatted evidence text."""
    return (
        f"TUYÊN BỐ CẦN KIỂM TRA:\n{claim}\n\n"
        f"BẰNG CHỨNG ỦNG HỘ (các điều khoản có thể liên quan):\n{support_text}\n\n"
        f"BẰNG CHỨNG PHỦ NHẬN (các điều khoản có thể mâu thuẫn):\n{contra_text}"
    )


def validate_direct_evidence(direct_evidence: str | None, support_text: str) -> str | None:
    if not direct_evidence:
        return None

    cited_articles = re.findall(r'Điều\s+\d+', direct_evidence, re.IGNORECASE)

    if cited_articles:
        for article in cited_articles:
            article_num = re.search(r'\d+', article).group()
            if (f"Điều {article_num}" in support_text or
                f"Điều {article_num}]" in support_text):
                return direct_evidence
        return None

    if direct_evidence.strip() and support_text.strip():
        evidence_words = set(
            w.lower() for w in re.findall(r'\w+', direct_evidence)
            if len(w) > 4
        )
        support_words = set(
            w.lower() for w in re.findall(r'\w+', support_text)
            if len(w) > 4
        )
        overlap = evidence_words & support_words
        if len(overlap) >= 3:
            return direct_evidence

    return None


def calculate_confidence(best_rerank: float, llm_confidence: float) -> float:
  
    return round((best_rerank * llm_confidence) ** 0.5, 4)


async def evaluate_final(
    claim: str,
    docs: list,
    contra_docs: Optional[list] = None,
) -> dict:
    if contra_docs is None:
        contra_docs = []

    if not docs and not contra_docs:
        return _build_fallback("Không tìm thấy tài liệu liên quan trong Database.")

    if not docs and contra_docs:
        best_rerank = max(
            (d.get("rerank_score", d.get("score", 0.0)) for d in contra_docs),
            default=0.0,
        )
        if best_rerank < 0.10:
            return _build_fallback(
                f"Độ liên quan của bằng chứng phủ nhận quá thấp ({round(best_rerank * 100, 1)}%)."
            )
    else:
        best_rerank = max(
            (d.get("rerank_score", d.get("score", 0.0)) for d in docs),
            default=0.0,
        )
        if best_rerank < 0.10:
            return _build_fallback(
                f"Độ liên quan của dữ liệu thô quá thấp ({round(best_rerank * 100, 1)}%)."
            )

    support_text = _format_docs(docs)
    contra_text  = _format_docs(contra_docs)

    try:
        user_msg = build_user_message(claim, support_text, contra_text)

        full_prompt = SYSTEM_INSTRUCTIONS + "\n" + user_msg
        word_count  = len(full_prompt.split())
        token_est   = int(word_count * 1.5)  
        print(f"[DEBUG] Prompt words={word_count}, est_tokens={token_est}, limit=4096")
        if token_est > 3500:
            print(f"[WARNING] Prompt approaching context limit! ({token_est} tokens)")

        client = ollama.AsyncClient()
        response = await client.chat(
            model   = "llama3.2",
            messages= [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user",   "content": user_msg},
            ],
            options = {
                "temperature":    0,
                "num_predict":    512,   
                "num_ctx":        4096,  
                "repeat_penalty": 1.1,  
                "seed":           42,
            },
            format  = "json",
        )

        raw_text = response["message"]["content"].strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            llm_res = json.loads(raw_text)
            llm_res["raw_llm_text"] = raw_text
        except Exception as parse_err:
            llm_res = {
                "verdict":         "INSUFFICIENT",
                "confidence":      0.3,
                "reasoning":       f"Lỗi phân tích JSON: {parse_err}",
                "direct_evidence": None,
            }

        if isinstance(llm_res.get("reasoning"), (dict, list)):
            llm_res["reasoning"] = json.dumps(llm_res["reasoning"], ensure_ascii=False)
        
        evidence_list = llm_res.get("evidence_used", [])
        if isinstance(evidence_list, list):
            raw_evidence = ", ".join(str(e) for e in evidence_list) if evidence_list else None
        else:
            raw_evidence = str(evidence_list) if evidence_list else None

        label = llm_res.get("label", "INSUFFICIENT")
        if label == "TRUE":
            verdict = "SUPPORTED"
        elif label == "FAKE":
            verdict = "CONTRADICTED"
        else:
            verdict = "INSUFFICIENT"

        llm_confidence = float(llm_res.get("confidence", 0.0))
        reasoning      = llm_res.get("reasoning", "AI không đưa ra lý giải cụ thể.")

        llm_confidence = max(0.0, min(1.0, llm_confidence))
        conf_final     = calculate_confidence(best_rerank, llm_confidence)

        if docs:
            validated_evidence = validate_direct_evidence(raw_evidence, support_text)
            if validated_evidence is None and verdict == "SUPPORTED":
                verdict        = "INSUFFICIENT"
                llm_confidence = min(llm_confidence, 0.45)
                conf_final     = calculate_confidence(best_rerank, llm_confidence)
                reasoning      = "[Bằng chứng không xác thực] " + reasoning
        else:
            validated_evidence = raw_evidence

        if verdict == "INSUFFICIENT" and conf_final > 0.0:
            conf_final = max(conf_final, 0.50)

        return {
            "verdict":         verdict,
            "confidence":      conf_final,
            "reasoning":       reasoning,
            "direct_evidence": validated_evidence,
            "evidence":        validated_evidence or "N/A",
            "reason":          reasoning,
            "sufficient":      conf_final >= 0.70 and verdict not in ("INSUFFICIENT", "ERROR"),
            "raw_llm_text":    raw_text,
        }

    except Exception as e:
        return _build_fallback(f"Lỗi phân tích: {str(e)}")


def _build_fallback(reason: str) -> dict:
    return {
        "verdict":         "INSUFFICIENT",
        "confidence":      0.0,
        "reasoning":       reason,
        "direct_evidence": None,
        "evidence":        "N/A",
        "reason":          reason,
        "sufficient":      False,
    }
