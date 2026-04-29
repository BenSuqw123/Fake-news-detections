"""
verdict_aggregator.py
=====================
Aggregates per-method evaluation results into a final truthfulness score & label.

Rules (in priority order):
  1. Instant FAKE  — any method returns CONTRADICTED
  2. Weighted score across methods (method_weight × verdict_weight × confidence)
  3. Threshold override — hybrid INSUFFICIENT: map confidence to fixed score bands
  4. Cap at 0.65   — score>=0.70 but hybrid lacks strong validated direct_evidence
  5. Labels        — REAL ≥ 0.70, UNCERTAIN ≥ 0.50, FAKE < 0.50
"""

VERDICT_WEIGHTS = {
    "SUPPORTED":    1.0,
    "PARTIAL":      0.5,
    "INSUFFICIENT": 0.2,
    "CONTRADICTED": 0.0,
    "ERROR":        0.0,
}

METHOD_WEIGHTS = {
    "hybrid_rrf":    0.60,
    "semantic_only": 0.25,
    "bm25_only":     0.15,
}


def compute_truthfulness_score(results: list, hybrid_direct_evidence: str = None) -> dict:
    """
    Parameters
    ----------
    results                : list of RetrievalResult (Pydantic objects with .method,
                             .verdict, .confidence attributes).
    hybrid_direct_evidence : direct_evidence string from the hybrid_rrf evaluation,
                             or None/empty if not available.

    Returns
    -------
    { "score": float, "label": str, "rule_applied": str }
    """

    # ── Rule 1: Instant FAKE on any CONTRADICTED result ───────────────────────
    for r in results:
        if r.verdict == "CONTRADICTED":
            return {
                "score":       round(0.10, 4),
                "label":       "FAKE",
                "rule_applied": (
                    f"Tìm thấy bằng chứng mâu thuẫn trực tiếp từ phương pháp '{r.method}'. "
                    "Tuyên bố bị bác bỏ."
                ),
            }

    # ── Rule 2: Weighted score ─────────────────────────────────────────────────
    total_score = 0.0
    for r in results:
        v_weight = VERDICT_WEIGHTS.get(r.verdict, 0.0)
        m_weight = METHOD_WEIGHTS.get(r.method,  0.0)
        total_score += m_weight * v_weight * r.confidence

    score = total_score

    # ── Rule 3: Handle INSUFFICIENT hybrid result ─────────────────────────────
    # When hybrid returns INSUFFICIENT the weighted formula collapses to ~0.08
    # (0.6 × 0.2 × conf), losing the LLM's nuanced confidence signal.
    # Use explicit thresholds instead so the score reflects how confident the
    # LLM was that evidence is absent.
    has_direct_evidence = bool(
        hybrid_direct_evidence
        and str(hybrid_direct_evidence).strip().lower() not in ("null", "none", "")
    )
    hybrid = next((r for r in results if r.method == "hybrid_rrf"), None)

    if hybrid:
        if hybrid.verdict == "INSUFFICIENT":
            llm_conf = hybrid.confidence
            if llm_conf >= 0.80:
                # LLM confident there's partial evidence → borderline UNCERTAIN
                score = 0.58
            elif llm_conf >= 0.60:
                # LLM moderately confident → FAKE borderline
                score = 0.44
            else:
                # LLM confident nothing exists → clear FAKE
                score = 0.25
        elif hybrid.verdict == "SUPPORTED" and not has_direct_evidence:
            # LLM claimed SUPPORTED but gave no specific clause → likely hallucination
            score = min(score, 0.48)

    # ── Rule 4: Require validated direct_evidence for REAL label ─────────────
    # Only allow REAL if hybrid explicitly returned SUPPORTED with high confidence
    # AND a validated direct_evidence article was found in the retrieved docs.
    has_strong_direct_evidence = (
        hybrid is not None
        and hybrid.verdict == "SUPPORTED"
        and hybrid.confidence >= 0.80
        and has_direct_evidence
    )

    if score >= 0.70 and not has_strong_direct_evidence:
        score = min(score, 0.65)

    # ── Rule 5: Labels ─────────────────────────────────────────────────────────
    if score >= 0.70:
        label        = "REAL"
        rule_applied = (
            f"Tìm thấy bằng chứng pháp lý trực tiếp ủng hộ tuyên bố "
            f"(Điểm: {score:.2f})."
        )
    elif score >= 0.50:
        label        = "UNCERTAIN"
        rule_applied = (
            f"Bằng chứng không đầy đủ hoặc chỉ gián tiếp "
            f"(Điểm: {score:.2f}). Cần chuyên gia xem xét thêm."
        )
    else:
        label        = "FAKE"
        rule_applied = (
            f"Thiếu bằng chứng pháp lý hoặc tuyên bố mâu thuẫn với quy định "
            f"(Điểm: {score:.2f})."
        )

    return {
        "score":       round(score, 4),
        "label":       label,
        "rule_applied": rule_applied,
    }
