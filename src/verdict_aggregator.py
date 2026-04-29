VERDICT_WEIGHTS = {
    "SUPPORTED":     1.0,
    "PARTIAL":       0.5,
    "INSUFFICIENT":  0.3,
    "CONTRADICTED":  0.0,
    "ERROR":         0.0,
}

METHOD_WEIGHTS = {
    "hybrid_rrf":    0.60, 
    "semantic_only": 0.25, 
    "bm25_only":     0.15, 
}

def compute_truthfulness_score(results: list) -> dict:
    """
    Weighted average of confidence scores across all 3 retrieval methods.
    Each method's contribution = method_weight * (verdict_weight * confidence)
    
    Rule-based label:
      score >= 0.70  → REAL    (high confidence supported)
      score >= 0.50  → UNCERTAIN (borderline, needs human review)
      score <  0.50  → FAKE    (contradicted or insufficient evidence)
    
    Return: { score, label, rule_applied }
    """
    total_score = 0.0
    for res in results:
        v_weight = VERDICT_WEIGHTS.get(res.verdict, 0.3)
        m_weight = METHOD_WEIGHTS.get(res.method, 0.0)
        total_score += m_weight * (v_weight * res.confidence)
        
    score = total_score
    if score >= 0.7:
        label = "REAL"
    elif score >= 0.50:
        label = "UNCERTAIN"
    else:
        label = "FAKE"
        
    if label == "REAL":
        rule_applied = "Hệ thống tìm thấy bằng chứng vững chắc ủng hộ tuyên bố (Điểm >= 70%)."
    elif label == "UNCERTAIN":
        rule_applied = "Bằng chứng không rõ ràng hoặc chỉ đúng một phần (50% <= Điểm < 70%). Cần chuyên gia xem xét thêm."
    else:
        rule_applied = "Bằng chứng bị bác bỏ hoặc không có cơ sở pháp lý đủ mạnh (Điểm < 50%)."
        
    return {
        "score": round(score, 4),
        "label": label,
        "rule_applied": rule_applied
    }
