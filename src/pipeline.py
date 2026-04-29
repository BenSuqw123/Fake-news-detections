import asyncio
import time
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from user_input_processing.Guardrail_input import check_input_validity
from user_input_processing.claimExtractor_getKeyword import extract_atomic_claims
from user_input_processing.evaluation_layer import evaluate_final
from src.chunking import chunk_query
from src.retriever.search_chromadb import search_chroma
from src.retriever.search_bm25 import search_bm25
from user_input_processing.rrf import reciprocal_rank_fusion
from user_input_processing.re_ranking import apply_reranking

async def async_process_retrieval(claim_text: str, keywords: str, method: str, top_k=5):
    """Executes the specific retrieval mode."""
    vector_docs = []
    bm25_docs = []
    
    if method in ["semantic_only", "hybrid_rrf"]:
        try:
            # search_chroma is synchronous but we can run it in a thread or just call it directly.
            vector_docs = search_chroma(claim_text, 20) or []
        except Exception as e:
            print(f"Chroma Error: {e}")
            
    if method in ["bm25_only", "hybrid_rrf"]:
        try:
            bm25_docs = search_bm25(keywords or claim_text, 20) or []
        except Exception as e:
            print(f"BM25 Error: {e}")

    if method == "hybrid_rrf":
        fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
    elif method == "semantic_only":
        fused = vector_docs
    else: # bm25_only
        fused = bm25_docs

    return apply_reranking(claim_text, fused, top_k=top_k)

async def process_single_claim(claim_data: dict, method: str):
    """Processes a single extracted claim."""
    claim_text = claim_data.get("claim", "")
    keywords = claim_data.get("keywords", "")
    
    docs = await async_process_retrieval(claim_text, keywords, method)
    result = await evaluate_final(claim_text, docs)
    result["claim"] = claim_text
    
    # Extract top docs safely
    top_docs_text = []
    for d in (docs or [])[:5]:
        meta = d.get("metadata", {})
        title = meta.get("law_title", "Văn bản")
        article = meta.get("article", "Điều khoản")
        content = d.get("text", "")
        top_docs_text.append(f"[{title} - {article}]: {content}")
        
    result["top_docs"] = top_docs_text
    return result

async def run_pipeline(user_input: str, retrieval_method: str = "hybrid_rrf", pre_extracted_claims: list = None, mode: str = "production") -> dict:
    """
    Unified entry point for processing.
    If pre_extracted_claims is provided, it skips Guardrail and Extraction (used for parallel routing).
    """
    start_time = time.time()
    all_claims = []
    
    if pre_extracted_claims is not None:
        all_claims = pre_extracted_claims
    else:
        # 1. Guardrail Input
        guardrail = await check_input_validity(user_input)
        if guardrail["status"] == "REJECT":
            return {"verdict": "ERROR", "confidence": 0.0, "reason": guardrail["message"], "method": retrieval_method, "top_docs": [], "processing_time_ms": 0}

        clean_query = guardrail.get("clean_query", user_input)
        
        # 2. Claim Extraction
        input_chunks = chunk_query(clean_query)
        for chunk in input_chunks:
            results = await extract_atomic_claims(chunk)
            all_claims.extend(results)
            
        if not all_claims:
            return {"verdict": "ERROR", "confidence": 0.0, "reason": "Không tìm thấy ý định pháp lý.", "method": retrieval_method, "top_docs": [], "processing_time_ms": 0}

    # 3. Process all claims concurrently
    tasks = [process_single_claim(c, retrieval_method) for c in all_claims]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    clean_results = []
    all_top_docs = []
    for r in results:
        if isinstance(r, Exception):
            clean_results.append({"verdict": "ERROR", "reason": str(r), "claim": "Lỗi Task", "confidence": "0.0%"})
        else:
            clean_results.append(r)
            all_top_docs.extend(r.get("top_docs", []))
            
    # 4. Synthesize Overall Verdict
    supported = sum(1 for r in clean_results if r.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for r in clean_results if r.get("verdict") == "CONTRADICTED")
    insufficient = sum(1 for r in clean_results if r.get("verdict") == "INSUFFICIENT")
    
    if contradicted > 0:
        verdict = "CONTRADICTED"
    elif supported == len(clean_results) and len(clean_results) > 0:
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIAL" if supported > 0 else "INSUFFICIENT"
        
    # Calculate confidence
    conf_scores = []
    for r in clean_results:
        try:
            conf = float(str(r.get("confidence", "0")).replace("%", "")) / 100.0
            conf_scores.append(conf)
        except:
            conf_scores.append(0.0)
            
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    output = {
        "method": retrieval_method,
        "verdict": verdict,
        "confidence": round(avg_conf, 2),
        "top_docs": list(dict.fromkeys(all_top_docs))[:5],  # Deduplicate & limit to 5
        "details": clean_results,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    }
    
    if mode == "eval":
        output["eval_metrics"] = {
            "claims_found": len(all_claims),
            "supported": supported,
            "contradicted": contradicted,
            "insufficient": insufficient
        }
        
    return output
