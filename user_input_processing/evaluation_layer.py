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
SYSTEM_INSTRUCTIONS = """Bạn là chuyên gia pháp lý Việt Nam với 20 năm kinh nghiệm.
Nhiệm vụ của bạn là đánh giá xem một tuyên bố có được xác nhận bởi các văn bản pháp luật hay không.

QUY TRÌNH BẮT BUỘC (thực hiện theo thứ tự, không bỏ qua bước nào):

Bước 1 — Đọc kỹ TUYÊN BỐ: Xác định chính xác tuyên bố đang nói về quyền/nghĩa vụ/quy định GÌ.

Bước 2 — Kiểm tra từng BẰNG CHỨNG ỦNG HỘ:
- Bằng chứng này có TRỰC TIẾP xác nhận đúng nội dung tuyên bố không?
- "Trực tiếp" = điều khoản đó nói rõ về đúng quyền/nghĩa vụ được đề cập trong tuyên bố
- Cùng chủ đề chung KHÔNG tính là bằng chứng trực tiếp
- Ví dụ: tuyên bố về "quyền lập đảng" → Điều 15 "quyền công dân" là KHÔNG trực tiếp

Bước 3 — Kiểm tra từng BẰNG CHỨNG PHỦ NHẬN:
- Bằng chứng này có TRỰC TIẾP mâu thuẫn với tuyên bố không?
- Ghi rõ: điều khoản nào mâu thuẫn và tại sao

Bước 4 — Kết luận theo đúng logic sau:
- Có bằng chứng ủng hộ TRỰC TIẾP + không có mâu thuẫn → SUPPORTED
- Có bằng chứng mâu thuẫn TRỰC TIẾP → CONTRADICTED
- Có bằng chứng ủng hộ một phần VÀ mâu thuẫn một phần → PARTIAL
- KHÔNG có bằng chứng nào trực tiếp xác nhận → INSUFFICIENT

LUẬT QUAN TRỌNG NHẤT:
- "Không tìm thấy bằng chứng phủ nhận" KHÔNG có nghĩa là tuyên bố đúng
- Chỉ SUPPORTED khi tìm được điều khoản NÓI RÕ về đúng nội dung tuyên bố
- Nếu tuyên bố về quyền X nhưng không tìm thấy điều khoản nào quy định quyền X → INSUFFICIENT
- Nếu tuyên bố sai tên chức danh, sai tên cơ quan, sai quy trình → CONTRADICTED
- Ví dụ: tuyên bố "công dân lập đảng độc lập" → Điều 7 (bầu cử/đầu phiếu) KHÔNG trực tiếp; phải tìm Điều 4 (vai trò lãnh đạo của Đảng Cộng sản) → CONTRADICTED
- Ví dụ: tuyên bố "quyền tự do ngôn luận" → Điều 25 Hiến pháp quy định rõ → SUPPORTED

Trả lời CHỈ bằng JSON, không thêm bất kỳ text nào khác:
{
  "verdict": "SUPPORTED|CONTRADICTED|PARTIAL|INSUFFICIENT",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<2-3 câu tiếng Việt giải thích: bằng chứng nào trực tiếp xác nhận/phủ nhận và tại sao>",
  "direct_evidence": "<tên điều khoản cụ thể VÀ trích dẫn ngắn nội dung, hoặc null nếu không có>"
}"""


def _format_docs(docs: list) -> str:
    """Convert doc-dict list to a labelled paragraph string for the prompt."""
    if not docs:
        return "(Không có bằng chứng)"
    lines = []
    for d in docs:
        if isinstance(d, dict):
            meta  = d.get("metadata") or {}
            title = meta.get("law_title", "Văn bản")
            art   = meta.get("article",   "?")
            text  = d.get("text", "")
            lines.append(f"[{title} - Điều {art}]: {text}")
        else:
            lines.append(str(d))
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
        client = ollama.AsyncClient()
        response = await client.chat(
            model   = "llama3.2",
            messages= [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user",   "content": build_user_message(claim, support_text, contra_text)},
            ],
            options = {"temperature": 0, "num_predict": 1024, "seed": 42},
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
