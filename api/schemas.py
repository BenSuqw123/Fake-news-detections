from pydantic import BaseModel
from typing import List

class VerificationRequest(BaseModel):
    text: str

class RetrievalResult(BaseModel):
    method: str
    top_docs: List[str]
    verdict: str
    confidence: float

class VerificationResult(BaseModel):
    input_text: str
    retrieval_comparison: List[RetrievalResult]
    final_verdict: str
    truthfulness_score: float
    label: str
    rule_applied: str
    processing_time_ms: int
