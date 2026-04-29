"""Quick test of Case 1 with proper Vietnamese text."""
import sys
import io
import asyncio
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

from underthesea import word_tokenize as _wt
_wt("khoi dong", format="text")


async def main():
    from src.pipeline import run_pipeline
    from src.verdict_aggregator import compute_truthfulness_score

    class R:
        def __init__(self, method, verdict, confidence):
            self.method = method
            self.verdict = verdict
            self.confidence = confidence

    # Case 1 with actual Vietnamese text (as the real API receives after Guardrail)
    claim_vi = {
        "claim": "Theo Điều 25 Hiến pháp năm 2013, công dân có quyền tự do ngôn luận, tự do báo chí.",
        "keywords": "",
    }

    print("Testing Case 1 with proper Vietnamese text...")
    result = await run_pipeline(
        user_input=claim_vi["claim"],
        retrieval_method="hybrid_rrf",
        pre_extracted_claims=[claim_vi],
    )

    v  = result.get("verdict")
    c  = float(result.get("confidence", 0.0))
    de = result.get("direct_evidence")
    rs = [R("hybrid_rrf", v, c)]
    sr = compute_truthfulness_score(rs, hybrid_direct_evidence=de)

    print(f"verdict        = {v}")
    print(f"confidence     = {c:.4f}")
    print(f"direct_evidence= {de}")
    print(f"score          = {sr['score']}")
    print(f"label          = {sr['label']}")
    details = result.get("details", [{}])
    print(f"reasoning      = {details[0].get('reasoning', '')[:120]}")


asyncio.run(main())
