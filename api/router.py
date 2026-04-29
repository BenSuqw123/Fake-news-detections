import time
from fastapi import APIRouter
from api.schemas import VerificationRequest, VerificationResult, RetrievalResult
from src.pipeline import run_pipeline
from src.verdict_aggregator import compute_truthfulness_score
from user_input_processing.Guardrail_input import check_input_validity
from user_input_processing.claimExtractor_getKeyword import extract_atomic_claims
from src.chunking import chunk_query

router = APIRouter()


def _format_doc_list(docs: list) -> list[str]:
    """Convert raw doc dicts to display strings. Handles dict or plain str."""
    out = []
    for d in (docs or []):
        if isinstance(d, dict):
            meta  = d.get("metadata") or {}
            title = meta.get("law_title", "Văn bản")
            art   = meta.get("article",   "?")
            text  = d.get("text", "")
            out.append(f"[{title} - Điều {art}]: {text}")
        else:
            out.append(str(d))
    return out


@router.post("/verify", response_model=VerificationResult)
async def verify_claim(request: VerificationRequest):
    start_time = time.time()

    # ── Step 1: Guardrail (run once, shared across all 3 methods) ─────────────
    guardrail = await check_input_validity(request.text)
    if guardrail["status"] == "REJECT":
        return VerificationResult(
            input_text            = request.text,
            retrieval_comparison  = [],
            final_verdict         = "ERROR",
            truthfulness_score    = 0.0,
            label                 = "REJECTED",
            rule_applied          = guardrail["message"],
            processing_time_ms    = int((time.time() - start_time) * 1000),
        )

    clean_query = guardrail.get("clean_query", request.text)

    # ── Step 2: Claim extraction (run once, shared) ───────────────────────────
    all_claims = []
    for chunk in chunk_query(clean_query):
        all_claims.extend(await extract_atomic_claims(chunk))

    if not all_claims:
        return VerificationResult(
            input_text            = request.text,
            retrieval_comparison  = [],
            final_verdict         = "ERROR",
            truthfulness_score    = 0.0,
            label                 = "ERROR",
            rule_applied          = "Không tìm thấy ý định pháp lý.",
            processing_time_ms    = int((time.time() - start_time) * 1000),
        )

    # ── Step 3: Run 3 retrieval modes sequentially ───────────────────────────
    # (CPU-only; sequential avoids competing Ollama requests)
    methods = ["bm25_only", "semantic_only", "hybrid_rrf"]
    raw_results = []
    for method in methods:
        result = await run_pipeline(
            request.text,
            method,
            pre_extracted_claims=all_claims,
        )
        raw_results.append(result)

    # ── Step 4: Build RetrievalResult objects (now with evidence fields) ──────
    retrieval_results = []
    for method, r in zip(methods, raw_results):
        if isinstance(r, Exception):
            retrieval_results.append(
                RetrievalResult(
                    method     = method,
                    top_docs   = [],
                    verdict    = "ERROR",
                    confidence = 0.0,
                )
            )
        else:
            retrieval_results.append(
                RetrievalResult(
                    method          = method,
                    top_docs        = r.get("top_docs", []),
                    verdict         = r.get("verdict", "INSUFFICIENT"),
                    confidence      = float(r.get("confidence", 0.0)),
                    support_docs    = _format_doc_list(r.get("support_docs", [])),
                    contra_docs     = _format_doc_list(r.get("contra_docs",  [])),
                    direct_evidence = r.get("direct_evidence"),
                    reasoning       = (r.get("details") or [{}])[0].get("reasoning"),
                )
            )

    # ── Step 5: Aggregate truthfulness score ──────────────────────────────────
    hybrid_result = next(
        (r for r in retrieval_results if r.method == "hybrid_rrf"), None
    )
    hybrid_direct_evidence = hybrid_result.direct_evidence if hybrid_result else None

    scoring_result = compute_truthfulness_score(
        retrieval_results,
        hybrid_direct_evidence=hybrid_direct_evidence,
    )

    # ── Step 6: Final verdict derived from label — single source of truth ─────
    label        = scoring_result["label"]
    all_verdicts = [r.verdict for r in retrieval_results]

    if label == "REAL":
        final_verdict = "SUPPORTED"
    elif label == "FAKE":
        final_verdict = "CONTRADICTED" if "CONTRADICTED" in all_verdicts else "INSUFFICIENT"
    elif label == "UNCERTAIN":
        final_verdict = "PARTIAL" if "PARTIAL" in all_verdicts else "INSUFFICIENT"
    else:
        final_verdict = "ERROR"

    return VerificationResult(
        input_text            = request.text,
        retrieval_comparison  = retrieval_results,
        final_verdict         = final_verdict,
        truthfulness_score    = scoring_result["score"],
        label                 = scoring_result["label"],
        rule_applied          = scoring_result["rule_applied"],
        processing_time_ms    = int((time.time() - start_time) * 1000),
    )
