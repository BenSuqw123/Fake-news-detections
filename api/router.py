import asyncio
import time
from fastapi import APIRouter
from api.schemas import VerificationRequest, VerificationResult, RetrievalResult
from src.pipeline import run_pipeline
from src.verdict_aggregator import compute_truthfulness_score
from user_input_processing.Guardrail_input import check_input_validity
from user_input_processing.claimExtractor_getKeyword import extract_atomic_claims
from src.chunking import chunk_query

router = APIRouter()

@router.post("/verify", response_model=VerificationResult)
async def verify_claim(request: VerificationRequest):
    start_time = time.time()
    
    # Pre-extract claims so we don't repeat the LLM call 3 times
    guardrail = await check_input_validity(request.text)
    if guardrail["status"] == "REJECT":
        # Return a rejected result
        return VerificationResult(
            input_text=request.text,
            retrieval_comparison=[],
            final_verdict="ERROR",
            truthfulness_score=0.0,
            label="REJECTED",
            rule_applied=guardrail["message"],
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        
    clean_query = guardrail.get("clean_query", request.text)
    input_chunks = chunk_query(clean_query)
    
    all_claims = []
    for chunk in input_chunks:
        results = await extract_atomic_claims(chunk)
        all_claims.extend(results)
        
    if not all_claims:
        return VerificationResult(
            input_text=request.text,
            retrieval_comparison=[],
            final_verdict="ERROR",
            truthfulness_score=0.0,
            label="ERROR",
            rule_applied="Không tìm thấy ý định pháp lý.",
            processing_time_ms=int((time.time() - start_time) * 1000)
        )

    # 1. Run 3 retrieval modes in parallel
    modes = ["bm25_only", "semantic_only", "hybrid_rrf"]
    tasks = [run_pipeline(request.text, mode, pre_extracted_claims=all_claims) for mode in modes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    retrieval_results = []
    for mode, r in zip(modes, results):
        if isinstance(r, Exception):
            retrieval_results.append(RetrievalResult(
                method=mode, top_docs=[], verdict="ERROR", confidence=0.0
            ))
        else:
            retrieval_results.append(RetrievalResult(
                method=mode,
                top_docs=r.get("top_docs", []),
                verdict=r.get("verdict", "INSUFFICIENT"),
                confidence=r.get("confidence", 0.0)
            ))
            
    # 2. Aggregate and Score
    scoring_result = compute_truthfulness_score(retrieval_results)
    
    # Find the most authoritative final verdict
    verdicts = [r.verdict for r in retrieval_results if r.verdict != "ERROR"]
    if "CONTRADICTED" in verdicts:
        final_verdict = "CONTRADICTED"
    elif "SUPPORTED" in verdicts:
        final_verdict = "SUPPORTED"
    elif "PARTIAL" in verdicts:
        final_verdict = "PARTIAL"
    else:
        final_verdict = "INSUFFICIENT"
        
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return VerificationResult(
        input_text=request.text,
        retrieval_comparison=retrieval_results,
        final_verdict=final_verdict,
        truthfulness_score=scoring_result["score"],
        label=scoring_result["label"],
        rule_applied=scoring_result["rule_applied"],
        processing_time_ms=processing_time_ms
    )
