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
# Trimmed to keep under ~350 tokens (Vietnamese chars count ~1.5× vs English)
SYSTEM_INSTRUCTIONS = """Bạn là chuyên gia pháp lý Việt Nam. Đánh giá tuyên bố dựa trên bằng chứng pháp luật.

QUY TRÌNH (theo thứ tự):
1. Đọc kỹ TUYÊN BỐ: xác định quyền/nghĩa vụ/quy định đang được đề cập.
2. Kiểm tra BẰNG CHỨNG ỦNG HỘ: có điều khoản NÓI RÕ về đúng nội dung tuyên bố không?
   - "Trực tiếp" = điều khoản quy định chính xác quyền/nghĩa vụ đó.
   - Cùng chủ đề chung KHÔNG tính là bằng chứng trực tiếp.
3. Kiểm tra BẰNG CHỨNG PHỦ NHẬN: điều khoản nào mâu thuẫn trực tiếp?
4. Kết luận:
   - Có bằng chứng ủng hộ trực tiếp, không có mâu thuẫn → SUPPORTED
   - Có bằng chứng mâu thuẫn trực tiếp → CONTRADICTED
   - Có cả hai → PARTIAL
   - Không có bằng chứng trực tiếp → INSUFFICIENT

LUẬT QUAN TRỌNG:
- KHÔNG tìm thấy bằng chứng phủ nhận ≠ tuyên bố đúng.
- Chỉ SUPPORTED khi tìm được điều khoản NÓI RÕ nội dung tuyên bố.
- Sai tên/chức danh/cơ quan → CONTRADICTED.

Trả lời CHỈ bằng JSON:
{
  "verdict": "SUPPORTED|CONTRADICTED|PARTIAL|INSUFFICIENT",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 câu tiếng Việt giải thích bằng chứng>",
  "direct_evidence": "<tên điều khoản cụ thể và trích dẫn ngắn, hoặc null>"
}"""


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
        except Exception as parse_err:
            llm_res = {
                "verdict":         "INSUFFICIENT",
                "confidence":      0.3,
                "reasoning":       f"Lỗi phân tích JSON: {parse_err}",
                "direct_evidence": None,
            }

        # Coerce fields — model occasionally returns nested dicts instead of strings
        if isinstance(llm_res.get("reasoning"), dict):
            llm_res["reasoning"] = str(llm_res["reasoning"])
        if isinstance(llm_res.get("direct_evidence"), dict):
            de_dict = llm_res["direct_evidence"]
            # Extract the first string value rather than stringifying the whole dict
            extracted = next((v for v in de_dict.values() if isinstance(v, str)), None)
            llm_res["direct_evidence"] = extracted if extracted else None

        verdict        = llm_res.get("verdict", "INSUFFICIENT")
        llm_confidence = float(llm_res.get("confidence", 0.0))
        reasoning      = llm_res.get("reasoning", "AI không đưa ra lý giải cụ thể.")
        raw_evidence   = llm_res.get("direct_evidence") or None

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
