from pydantic import BaseModel
from typing import List, Optional


class VerificationRequest(BaseModel):
    text: str


class RetrievalResult(BaseModel):
    method:          str
    top_docs:        List[str]
    verdict:         str
    confidence:      float
    # NEW — dual evidence fields (optional for backward compat)
    support_docs:    Optional[List[str]] = None   # formatted supporting clauses
    contra_docs:     Optional[List[str]] = None   # formatted contradicting clauses
    direct_evidence: Optional[str]       = None   # specific clause that proves/disproves
    reasoning:       Optional[str]       = None   # LLM's chain-of-thought


class VerificationResult(BaseModel):
    input_text:             str
    retrieval_comparison:   List[RetrievalResult]
    final_verdict:          str
    truthfulness_score:     float
    label:                  str
    rule_applied:           str
    processing_time_ms:     int
