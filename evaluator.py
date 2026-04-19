import sys
import asyncio
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from user_input_processing.claim_extractor import extract_atomic_claims
from user_input_processing.check_input import (
    async_extract_key_info,
    normalize_bm25_query,
    reciprocal_rank_fusion,
    apply_reranking,
    HYBRID_CANDIDATE_K,
    RERANK_TOP_K
)
from RAG_LAW.Tranformer.search_chormadb import search_chroma
from RAG_LAW.Tranformer.search_bm25 import search_bm25
from user_input_processing.evaluation_layer import evaluate_evidence

async def run_full_pipeline(input_text: str) -> dict:
    """
    Runs the full end-to-end testing pipeline.
    This includes claim extraction, query transformation, hybrid retrieval, reranking, and evaluation.
    Logs intermediate outputs heavily for debugging.
    """
    print("\n" + "="*60)
    print("🚀 PIPELINE EVALUATION START")
    print("="*60)
    
    start_time = time.time()
    
    pipeline_result = {
        "input": input_text,
        "claims": []
    }
    
    # ---------------------------------------------------------
    # 1. EXTRACT CLAIMS
    # ---------------------------------------------------------
    print("\n[STAGE 1: CLAIM EXTRACTION]")
    try:
        t0 = time.time()
        claims = await extract_atomic_claims(input_text)
        t_extractor = time.time() - t0
        print(f"⏱️  Extraction Time: {t_extractor:.2f}s")
        print(f"✅ Found {len(claims)} claims.")
        for i, c in enumerate(claims):
            print(f"  - Claim {i+1}: {c['claim']} (Verifiable: {c.get('is_verifiable', False)})")
    except Exception as e:
        print(f"❌ Error extracting claims: {e}")
        pipeline_result["error"] = f"Claim extraction failed: {str(e)}"
        return pipeline_result

    # ---------------------------------------------------------
    # PROCESS EACH CLAIM
    # ---------------------------------------------------------
    for i, c in enumerate(claims):
        claim_text = c['claim']
        if not c.get('is_verifiable'):
            print(f"\n⏭️  Skipping Claim {i+1} (Not Verifiable): {claim_text}")
            continue
            
        print("\n" + "-"*60)
        print(f"🔍 PROCESSING CLAIM {i+1}: {claim_text}")
        print("-"*60)
        
        claim_result = {
            "claim": claim_text,
            "retrieved_docs": [],
            "reranked_docs": [],
            "evaluation": {}
        }
        
        try:
            # ---------------------------------------------------------
            # 2. QUERY TRANSFORMATION
            # ---------------------------------------------------------
            print("\n[STAGE 2: QUERY TRANSFORMATION]")
            t0 = time.time()
            clean_query = await async_extract_key_info(claim_text)
            vector_query = f"{claim_text}, {clean_query}"
            bm25_query = normalize_bm25_query(claim_text)
            t_query = time.time() - t0
            
            print(f"⏱️  Query Transform Time: {t_query:.2f}s")
            print(f"  - Extracted Keys: {clean_query}")
            print(f"  - Vector Query: {vector_query}")
            print(f"  - BM25 Query:   {bm25_query}")
            
            # ---------------------------------------------------------
            # 3. RETRIEVAL (DENSE + BM25 + RRF)
            # ---------------------------------------------------------
            print("\n[STAGE 3: HYBRID RETRIEVAL & RRF]")
            t0 = time.time()
            vector_docs = await asyncio.to_thread(search_chroma, vector_query, HYBRID_CANDIDATE_K)
            bm25_docs = await asyncio.to_thread(search_bm25, bm25_query, HYBRID_CANDIDATE_K)

            # Normalize scores for RRF
            for doc in vector_docs:
                doc["normalized_score"] = max(1.0 - doc["score"], 0.0)

            for doc in bm25_docs:
                doc["normalized_score"] = min(doc["score"] / 20.0, 1.0)
                
            fused_docs = reciprocal_rank_fusion(
                [vector_docs, bm25_docs],
                top_k=HYBRID_CANDIDATE_K,
            )
            t_retrieval = time.time() - t0
            
            print(f"⏱️  Retrieval Time: {t_retrieval:.2f}s")
            print(f"✅ Found {len(fused_docs)} fused documents.")
            print("  Top 3 Retrieved Docs (Before Rerank):")
            for j, doc in enumerate(fused_docs[:3]):
                snippet = doc.get("text", "")[:150].replace('\n', ' ')
                print(f"    {j+1}. [RRF: {doc.get('rrf_score', 0):.4f}] {snippet}...")
                
            claim_result["retrieved_docs"] = [
                {"id": d.get("id"), "rrf_score": round(d.get("rrf_score", 0), 4), "text": d.get("text", "")[:200]} 
                for d in fused_docs[:10]
            ]
            
            # ---------------------------------------------------------
            # 4. RERANKING
            # ---------------------------------------------------------
            print("\n[STAGE 4: RERANKING]")
            t0 = time.time()
            reranked_docs = apply_reranking(vector_query, fused_docs, top_k=RERANK_TOP_K)
            t_rerank = time.time() - t0
            
            print(f"⏱️  Reranking Time: {t_rerank:.2f}s")
            print("  Top Reranked Docs:")
            
            avg_score = 0.0
            for j, doc in enumerate(reranked_docs):
                score = doc.get("rerank_score", doc.get("normalized_score", 0))
                avg_score += score
                snippet = doc.get("text", "")[:150].replace('\n', ' ')
                print(f"    {j+1}. [Score: {score:.4f}] {snippet}...")
                
            if reranked_docs:
                print(f"  📊 Avg Rerank Score (Top {len(reranked_docs)}): {avg_score/len(reranked_docs):.4f}")
                
            claim_result["reranked_docs"] = [
                {"id": d.get("id"), "rerank_score": round(d.get("rerank_score", d.get("normalized_score", 0)), 4), "text": d.get("text", "")[:200]} 
                for d in reranked_docs
            ]
            
            # ---------------------------------------------------------
            # 5. EVALUATION LAYER
            # ---------------------------------------------------------
            print("\n[STAGE 5: VERIFICATION & EVALUATION]")
            t0 = time.time()
            eval_res = await evaluate_evidence(claim_text, reranked_docs)
            t_eval = time.time() - t0
            
            print(f"⏱️  Evaluation Time: {t_eval:.2f}s")
            print(f"  - Verdict:     {eval_res.get('verdict')}")
            print(f"  - Confidence:  {eval_res.get('confidence')}")
            print(f"  - Sufficient:  {eval_res.get('sufficient')}")
            print(f"  - Coverage:    {eval_res.get('coverage_score')}")
            print(f"  - Agreement:   {eval_res.get('agreement_score')}")
            print(f"  - Conflicts:   {eval_res.get('conflict_detected')}")
            print(f"  - Reason:      {eval_res.get('reason')}")
            
            claim_result["evaluation"] = eval_res
            
        except Exception as e:
            print(f"❌ Error processing claim '{claim_text}': {e}")
            claim_result["error"] = str(e)
            
        pipeline_result["claims"].append(claim_result)

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"🏁 PIPELINE EVALUATION COMPLETE (Total Time: {total_time:.2f}s)")
    print("="*60)
    
    return pipeline_result

if __name__ == "__main__":
    test_input = "Hiến pháp 2013 quy định Việt Nam là một quốc gia độc lập. Đồng thời cũng cho phép thiết lập các khu tự trị không tuân theo pháp luật chung."
    
    # Create windows event loop policy to avoid "Event loop is closed" errors when finishing on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    print("Testing Evaluator Setup...\n")
    try:
        result = asyncio.run(run_full_pipeline(test_input))
        print("\n" + "="*60)
        print("FINAL STRUCTURED OUTPUT (JSON):")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"Pipeline crashed entirely: {str(e)}")
