"""
evaluation_layer.py
===================
LLM-based claim evaluation using supporting AND contradicting evidence.

Key design choices:
- System/user message split: static instructions in system role, dynamic data in user role.
- direct_evidence is validated against retrieved docs to prevent hallucination.
- Confidence is the LLM's own 0-1 score grounded by best_rerank (geometric mean).
- evaluate_final() accepts optional contra_docs (default [] for backward compat).
"""
import re
import json
import ollama
import asyncio
from typing import Optional

# ── System message — static instructions only, no claim data ──────────────────
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

### Ví dụ 4 (TRUE - logic gián tiếp)
Tuyên bố: Công dân tham gia gián tiếp vào việc chọn Chủ tịch nước
→ TRUE
Lý do: Họ bầu ra đại biểu, và đại biểu bầu ra Chủ tịch nước.

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


# Max chars per doc snippet sent to LLM — keeps prompt inside 4096-token window
_DOC_CHAR_LIMIT = 250


def _format_docs(docs: list) -> str:
    """Convert doc-dict list to a labelled paragraph string for the prompt.

    Each doc text is truncated to _DOC_CHAR_LIMIT characters to keep the
    total prompt size within llama3.2's 4096-token context window.
    """
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
    """
    Verify that the LLM-cited evidence actually appears in the retrieved docs.
    Prevents hallucination from bypassing the INSUFFICIENT safety check.
    """
    if not direct_evidence:
        return None

    cited_articles = re.findall(r'Điều\s+\d+', direct_evidence, re.IGNORECASE)

    if not cited_articles:
        return None

    for article in cited_articles:
        article_num = re.search(r'\d+', article).group()
        if f"Điều {article_num}" in support_text or f"Điều {article_num}]" in support_text:
            return direct_evidence

    return None


def calculate_confidence(best_rerank: float, llm_confidence: float) -> float:
    """
    Ground the LLM's self-reported confidence with retrieval quality.
    Uses geometric mean so either factor = 0 collapses the score.
    """
    return round((best_rerank * llm_confidence) ** 0.5, 4)


async def evaluate_final(
    claim: str,
    docs: list,
    contra_docs: Optional[list] = None,
) -> dict:
    """
    Evaluate a single claim against supporting + contradicting evidence.

    Parameters
    ----------
    claim       : The claim string to evaluate.
    docs        : Supporting docs (list of dicts with 'text', 'metadata', 'rerank_score').
    contra_docs : Contradicting docs (same format; may be empty or raw strings).
    """
    if contra_docs is None:
        contra_docs = []

    # ── 1. Early exit when no docs at all ─────────────────────────────────────
    if not docs:
        return _build_fallback("Không tìm thấy tài liệu liên quan trong Database.")

    best_rerank = max(
        (d.get("rerank_score", d.get("score", 0.0)) for d in docs),
        default=0.0,
    )

    if best_rerank < 0.10:
        return _build_fallback(
            f"Độ liên quan của dữ liệu thô quá thấp ({round(best_rerank * 100, 1)}%)."
        )

    # ── 2. Format docs (computed once; reused for both prompt and validation) ──
    support_text = _format_docs(docs)
    contra_text  = _format_docs(contra_docs)

    # ── 3. Call LLM with system/user split ────────────────────────────────────
    try:
        user_msg = build_user_message(claim, support_text, contra_text)

        # ── Debug: estimate prompt token count before every Ollama call ────────
        full_prompt = SYSTEM_INSTRUCTIONS + "\n" + user_msg
        word_count  = len(full_prompt.split())
        token_est   = int(word_count * 1.5)   # Vietnamese inflates ~1.5× vs English
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
                "num_predict":    512,    # max output tokens
                "num_ctx":        4096,   # explicit context window — no ambiguity
                "repeat_penalty": 1.1,   # reduces repetition loops
                "seed":           42,
            },
            format  = "json",
        )

        # ── 4. Safe JSON parse — strip markdown fences, coerce field types ──────
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

        # Coerce fields — model occasionally returns nested dicts instead of strings
        if isinstance(llm_res.get("reasoning"), (dict, list)):
            import json
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

        # ── 5. Validate direct_evidence against retrieved docs ────────────────
        validated_evidence = validate_direct_evidence(raw_evidence, support_text)

        if validated_evidence is None and verdict == "SUPPORTED":
            verdict        = "INSUFFICIENT"
            llm_confidence = min(llm_confidence, 0.45)
            conf_final     = calculate_confidence(best_rerank, llm_confidence)
            reasoning      = "[Bằng chứng không xác thực] " + reasoning

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
