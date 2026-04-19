import json
import ollama
import asyncio

EVALUATION_SYSTEM_PROMPT = """You are the Evidential Evaluation Layer of a strict Fact-Checking System.
Your ONLY job is to evaluate if the provided array of retrieved DOCUMENTS contains sufficient information to verify the CLAIM.

INSTRUCTIONS:
1. COMPLETENESS: Determine if the documents contain all necessary elements to address the exact scope of the claim.
2. EVIDENCE: Classify the relationship:
   - SUPPORTED: Documents explicitly confirm the claim.
   - CONTRADICTED: Documents explicitly refute the claim.
   - INSUFFICIENT: Documents are irrelevant or lack specific details to conclude.
   - PARTIAL: Documents confirm part of the claim but are missing critical components.
3. CONSISTENCY: Check if any two documents provide opposing facts regarding the claim.

You must score `agreement_score` (0.0 to 1.0) based on how unified the documents are (if documents disagree on the core fact, score drops).
You must score `coverage_score` (0.0 to 1.0) based on how much of the claim is addressed by the documents (e.g., if a claim has 2 parts and only 1 is supported, score is 0.5).

OUTPUT FORMAT:
Return ONLY valid JSON. Absolutely no markdown blocks, no think tags, no text outside the JSON.
{
  "sufficient": true,
  "verdict": "SUPPORTED",
  "agreement_score": 1.0,
  "coverage_score": 1.0,
  "conflict_detected": false,
  "reason": "String detailing exactly which document IDs support/contradict/conflict."
}"""

def calculate_final_confidence(llm_eval_json: dict) -> float:
    """
    Algorithmically derives the final confidence score based on the LLM's raw graded metrics.
    """
    base_confidence = float(llm_eval_json.get("coverage_score", 0.0))
    
    # Penalty for document conflicts
    if llm_eval_json.get("conflict_detected", False):
        base_confidence *= float(llm_eval_json.get("agreement_score", 1.0))
        
    verdict = llm_eval_json.get("verdict", "INSUFFICIENT")
    
    # Penalty for partial verdicts
    if verdict == "PARTIAL":
        base_confidence *= 0.7 
        
    # Insufficient immediately zeroes confidence
    if verdict == "INSUFFICIENT":
        base_confidence = 0.0
        
    # Clamp between 0 and 1
    return max(0.0, min(1.0, round(base_confidence, 2)))

async def evaluate_evidence(claim: str, docs: list) -> dict:
    """
    Evaluates the quality of retrieved documents against an atomic claim.
    """
    # Pre-Check: Do we even have any minimally relevant documents?
    if not docs:
        return _build_fallback("No documents provided.")
        
    # Optional: If your pipeline provides 'rerank_score' or 'normalized_score', check it.
    best_score = max((doc.get("rerank_score", doc.get("normalized_score", 0.0)) for doc in docs), default=0.0)
    
    # If the absolute best document is statistically terrible, don't waste LLM tokens.
    if best_score < 0.20:
         return _build_fallback("Pre-check failed: Top retrieved document score is extremely low.")

    # Format context for LLM
    formatted_docs = []
    for i, doc in enumerate(docs):
        text = doc.get("text", "")
        # Add metadata context if available to help the LLM identify the source
        meta = doc.get("metadata", {})
        source_title = meta.get("law_title", meta.get("title", f"Doc_{i+1}"))
        article = meta.get("article", "")
        
        formatted_docs.append(f"[Document ID: {i+1} | Source: {source_title} {article}]\n{text}\n")
    
    documents_context = "\n".join(formatted_docs)
    
    user_prompt = f"CLAIM: \"{claim}\"\n\nDOCUMENTS:\n{documents_context}"
    
    try:
        client = ollama.AsyncClient()
        response = await client.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': EVALUATION_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0},
            format='json'
        )
        
        raw_output = response['message']['content'].strip()
        result = json.loads(raw_output)
        
        # Calculate final algorithmic confidence
        result["confidence"] = calculate_final_confidence(result)
        
        # Enforce strict sufficiency boolean constraint
        cond_sufficient = result.get("coverage_score", 0.0) >= 0.8 and result.get("verdict") in ["SUPPORTED", "CONTRADICTED"]
        result["sufficient"] = bool(cond_sufficient)
        
        return result
        
    except json.JSONDecodeError:
        return _build_fallback("LLM failed to output valid JSON.")
    except Exception as e:
        return _build_fallback(f"Evaluation request failed: {str(e)}")

def _build_fallback(reason: str) -> dict:
    return {
        "sufficient": False,
        "verdict": "INSUFFICIENT",
        "confidence": 0.0,
        "agreement_score": 0.0,
        "coverage_score": 0.0,
        "conflict_detected": False,
        "reason": reason
    }

if __name__ == "__main__":
    # Internal Unit Test Execution
    async def run_test():
        test_claim = "Tốc độ tối đa trong khu đông dân cư là 60 km/h và áp dụng cho cả xe máy, ô tô tải."
        
        test_docs = [
            {
                "text": "Xe mô tô (xe máy) khi chạy trong khu đông dân cư được phép chạy tối đa 40 km/h.",
                "metadata": {"title": "Luật Giao thông Đường bộ"},
                "rerank_score": 0.85
            },
            {
                "text": "Ô tô tải lưu thông khu vực đông dân cư có tốc độ tối đa là 60 km/h.",
                "metadata": {"title": "Luật Giao thông Đường bộ"},
                "rerank_score": 0.90
            }
        ]
        
        print(f"Testing Claim: {test_claim}\n")
        print("Running Evaluation Layer...")
        print("-" * 50)
        
        result = await evaluate_evidence(test_claim, test_docs)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(run_test())
