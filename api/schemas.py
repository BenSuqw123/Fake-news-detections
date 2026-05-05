from pydantic import BaseModel
from typing import List, Optional


class VerificationRequest(BaseModel):
    text: str


class RetrievalResult(BaseModel):
    method:          str
    top_docs:        List[str]
    verdict:         str
    confidence:      float
    support_docs:    Optional[List[str]] = None  
    contra_docs:     Optional[List[str]] = None  
    direct_evidence: Optional[str]       = None  
    reasoning:       Optional[str]       = None   


class VerificationResult(BaseModel):
    input_text:             str
    retrieval_comparison:   List[RetrievalResult]
    final_verdict:          str
    truthfulness_score:     float
    label:                  str
    rule_applied:           str
    processing_time_ms:     int


class QuickCheckResult(BaseModel):
    label:               str
    final_verdict:       str
    truthfulness_score:  float
    reasoning:           Optional[str] = None
    direct_evidence:     Optional[str] = None
    rule_applied:        str
    processing_time_ms:  int
