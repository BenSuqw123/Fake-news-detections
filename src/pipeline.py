"""
pipeline.py
===========
Full RAG pipeline: Guardrail → Claim extraction → Dual retrieval
(supporting + contradicting) → Reranking → LLM evaluation.
"""
import asyncio
import time
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import TOP_K_RETRIEVAL, TOP_K_RERANK
from user_input_processing.Guardrail_input import check_input_validity
from user_input_processing.claimExtractor_getKeyword import extract_atomic_claims
from user_input_processing.evaluation_layer import evaluate_final
from src.chunking import chunk_query
from src.retriever.search_chromadb import search_chroma, search_chroma_contradiction
from src.retriever.search_bm25 import search_bm25, search_bm25_contradiction
from user_input_processing.rrf import reciprocal_rank_fusion
from user_input_processing.re_ranking import apply_reranking


def _dedup_by_id(docs: list) -> list:
    """Remove duplicate docs (same 'id') keeping first occurrence."""
    seen = set()
    out  = []
    for d in docs:
        if isinstance(d, dict):
            doc_id = d.get("id")
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                out.append(d)
        else:
            out.append(d)   # raw strings — keep as-is
    return out


async def async_process_retrieval(claim_text: str, keywords: str, method: str) -> dict:
    """
    Returns { "support_docs": [...], "contra_docs": [...] }

    Both doc lists use the standard dict format:
      { id, text, metadata, score, method }
    """
    loop = asyncio.get_event_loop()
    vector_docs  = []
    bm25_docs    = []
    vector_contra = []
    bm25_contra  = []

    # ── Supporting retrieval ───────────────────────────────────────────────────
    if method in ("semantic_only", "hybrid_rrf"):
        try:
            vector_docs = await loop.run_in_executor(
                None, search_chroma, claim_text, TOP_K_RETRIEVAL
            ) or []
        except Exception as e:
            print(f"Chroma error: {e}")

    if method in ("bm25_only", "hybrid_rrf"):
        try:
            bm25_docs = await loop.run_in_executor(
                None, search_bm25, keywords or claim_text, TOP_K_RETRIEVAL
            ) or []
        except Exception as e:
            print(f"BM25 error: {e}")

    # ── Contradicting retrieval (NEW) ─────────────────────────────────────────
    if method in ("semantic_only", "hybrid_rrf"):
        try:
            vector_contra = await loop.run_in_executor(
                None, search_chroma_contradiction, claim_text, 5
            ) or []
        except Exception as e:
            print(f"Chroma contradiction error: {e}")

    if method in ("bm25_only", "hybrid_rrf"):
        try:
            bm25_contra = await loop.run_in_executor(
                None, search_bm25_contradiction, keywords or claim_text, 5
            ) or []
        except Exception as e:
            print(f"BM25 contradiction error: {e}")

    # ── Fuse supporting docs ──────────────────────────────────────────────────
    if method == "hybrid_rrf":
        fused = reciprocal_rank_fusion([vector_docs, bm25_docs])
    elif method == "semantic_only":
        fused = vector_docs
    else:
        fused = bm25_docs

    # ── Rerank supporting docs ────────────────────────────────────────────────
    reranked = await loop.run_in_executor(
        None, apply_reranking, claim_text, fused, TOP_K_RERANK
    )

    # ── Deduplicate contradicting docs ────────────────────────────────────────
    contra_docs = _dedup_by_id(vector_contra + bm25_contra)

    return {
        "support_docs": reranked or [],
        "contra_docs":  contra_docs,
    }


async def process_single_claim(claim_data: dict, method: str) -> dict:
    claim_text = claim_data.get("claim", "")
    keywords   = claim_data.get("keywords", "")

    retrieval  = await async_process_retrieval(claim_text, keywords, method)
    support    = retrieval["support_docs"]
    contra     = retrieval["contra_docs"]

    result = await evaluate_final(claim_text, support, contra_docs=contra)
    result["claim"] = claim_text

    # ── Format top_docs for display ───────────────────────────────────────────
    top_docs_text = []
    for d in (support or [])[:TOP_K_RERANK]:
        meta    = d.get("metadata", {})
        title   = meta.get("law_title", "Văn bản")
        article = meta.get("article",   "Điều khoản")
        content = d.get("text", "")
        top_docs_text.append(f"[{title} - {article}]: {content}")

    result["top_docs"]    = top_docs_text
    result["support_docs_raw"] = support
    result["contra_docs_raw"]  = contra

    return result


async def run_pipeline(
    user_input: str,
    retrieval_method: str = "hybrid_rrf",
    pre_extracted_claims: list = None,
    mode: str = "production",
) -> dict:
    """
    Unified entry point.
    Pass pre_extracted_claims to skip Guardrail + Extraction (used by router).
    """
    start_time = time.time()
    all_claims = []

    if pre_extracted_claims is not None:
        all_claims = pre_extracted_claims
    else:
        guardrail = await check_input_validity(user_input)
        if guardrail["status"] == "REJECT":
            return {
                "verdict":            "ERROR",
                "confidence":         0.0,
                "reason":             guardrail["message"],
                "method":             retrieval_method,
                "top_docs":           [],
                "processing_time_ms": 0,
            }

        clean_query = guardrail.get("clean_query", user_input)
        for chunk in chunk_query(clean_query):
            all_claims.extend(await extract_atomic_claims(chunk))

        if not all_claims:
            return {
                "verdict":            "ERROR",
                "confidence":         0.0,
                "reason":             "Không tìm thấy ý định pháp lý.",
                "method":             retrieval_method,
                "top_docs":           [],
                "processing_time_ms": 0,
            }

    tasks   = [process_single_claim(c, retrieval_method) for c in all_claims]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    clean_results = []
    all_top_docs  = []
    all_support   = []
    all_contra    = []
    direct_ev     = None   # take direct_evidence from first SUPPORTED claim

    for r in results:
        if isinstance(r, Exception):
            clean_results.append({
                "verdict":    "ERROR",
                "reason":     str(r),
                "claim":      "Lỗi Task",
                "confidence": 0.0,
            })
        else:
            clean_results.append(r)
            all_top_docs.extend(r.get("top_docs", []))
            all_support.extend(r.get("support_docs_raw", []))
            all_contra.extend(r.get("contra_docs_raw", []))
            if direct_ev is None and r.get("direct_evidence"):
                direct_ev = r["direct_evidence"]

    # ── Aggregate verdict ─────────────────────────────────────────────────────
    supported    = sum(1 for r in clean_results if r.get("verdict") == "SUPPORTED")
    contradicted = sum(1 for r in clean_results if r.get("verdict") == "CONTRADICTED")
    insufficient = sum(1 for r in clean_results if r.get("verdict") == "INSUFFICIENT")

    if contradicted > 0:
        verdict = "CONTRADICTED"
    elif supported == len(clean_results) and len(clean_results) > 0:
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIAL" if supported > 0 else "INSUFFICIENT"

    # ── Aggregate confidence ──────────────────────────────────────────────────
    conf_scores = []
    for r in clean_results:
        try:
            raw = r.get("confidence", 0.0)
            # Handle legacy "82.5%" string format as well as plain float
            conf_scores.append(
                float(str(raw).replace("%", "")) / 100.0
                if "%" in str(raw)
                else float(raw)
            )
        except Exception:
            conf_scores.append(0.0)

    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    output = {
        "method":             retrieval_method,
        "verdict":            verdict,
        "confidence":         round(avg_conf, 4),
        "direct_evidence":    direct_ev,
        "top_docs":           list(dict.fromkeys(all_top_docs))[:TOP_K_RERANK],
        "support_docs":       _dedup_by_id(all_support),
        "contra_docs":        _dedup_by_id(all_contra),
        "details":            clean_results,
        "processing_time_ms": int((time.time() - start_time) * 1000),
    }

    if mode == "eval":
        output["eval_metrics"] = {
            "claims_found": len(all_claims),
            "supported":    supported,
            "contradicted": contradicted,
            "insufficient": insufficient,
        }

    return output
