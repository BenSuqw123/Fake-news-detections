
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

    total_score = 0.0
    for r in results:
        v_weight = VERDICT_WEIGHTS.get(r.verdict, 0.0)
        m_weight = METHOD_WEIGHTS.get(r.method,  0.0)
        total_score += m_weight * v_weight * r.confidence

    score = total_score

    has_direct_evidence = bool(
        hybrid_direct_evidence
        and str(hybrid_direct_evidence).strip().lower() not in ("null", "none", "")
    )
    hybrid = next((r for r in results if r.method == "hybrid_rrf"), None)

    if hybrid:
        if hybrid.verdict == "INSUFFICIENT":
            llm_conf = hybrid.confidence
            if llm_conf >= 0.80:
                score = 0.58
            elif llm_conf >= 0.60:
                score = 0.44
            else:
                score = 0.25
        elif hybrid.verdict == "SUPPORTED" and not has_direct_evidence:
            score = min(score, 0.48)

    has_strong_direct_evidence = (
        hybrid is not None
        and hybrid.verdict == "SUPPORTED"
        and hybrid.confidence >= 0.80
        and has_direct_evidence
    )

    if score >= 0.70 and not has_strong_direct_evidence:
        score = min(score, 0.65)

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
