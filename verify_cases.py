"""
verify_cases.py
===============
Verification of 3 test cases for the dual-retrieval fix.
Uses pre_extracted_claims to bypass the LLM Guardrail so we can test
the retrieval+evaluation pipeline directly with Vietnamese text.

Run: python verify_cases.py
"""
import sys
import io
import asyncio
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent))

# Pre-warm underthesea in main thread (avoid race on executor threads)
from underthesea import word_tokenize as _wt
_wt("khoi dong", format="text")
print("[Pre-warm] underthesea OK")

# ── Test cases ────────────────────────────────────────────────────────────────
# Pre-extracted so we bypass Guardrail + ClaimExtractor (both use Ollama too)
# and measure purely the retrieval + evaluation quality.
CASES = [
    {
        "name":           "Case 1 — DUNG (expect REAL)",
        "claims":         [{"claim": "Theo Dieu 25 Hien phap 2013, cong dan co quyen tu do ngon luan, tu do bao chi, tiep can thong tin, hoi hop, lap hoi, bieu tinh.", "keywords": ""}],
        "expected_label": "REAL",
    },
    {
        "name":           "Case 2 — SAI (expect FAKE)",
        "claims":         [{"claim": "Cong dan Viet Nam co quyen tu do thanh lap dang phai chinh tri doc lap va ung cu tong thong theo hinh thuc pho thong dau phieu truc tiep theo Hien phap 2013.", "keywords": ""}],
        "expected_label": "FAKE",
    },
    {
        "name":           "Case 3 — MO HO (expect UNCERTAIN)",
        "claims":         [{"claim": "Luat Lao dong Viet Nam quy dinh nguoi lao dong duoc nghi toi thieu 15 ngay phep nam va nhan luong thang 13 bat buoc tu nguoi su dung lao dong.", "keywords": ""}],
        "expected_label": "UNCERTAIN",
    },
]


class _FakeResult:
    """Minimal stand-in for Pydantic RetrievalResult."""
    def __init__(self, method: str, verdict: str, confidence: float):
        self.method     = method
        self.verdict    = verdict
        self.confidence = confidence


async def main():
    from src.pipeline import run_pipeline
    from src.verdict_aggregator import compute_truthfulness_score

    print("\n" + "=" * 65)
    print("VERIFICATION — 3 test cases (dual-retrieval fix)")
    print("=" * 65)

    passed = 0

    for case in CASES:
        print(f"\n>> {case['name']}")
        try:
            # Skip guardrail by passing pre_extracted_claims
            result = await run_pipeline(
                user_input           = case["claims"][0]["claim"],
                retrieval_method     = "hybrid_rrf",
                pre_extracted_claims = case["claims"],
                mode                 = "production",
            )

            verdict = result.get("verdict", "?")
            conf    = float(result.get("confidence", 0.0))
            de      = result.get("direct_evidence")

            # Run the aggregator (single method — hybrid only for unit test)
            rs        = [_FakeResult("hybrid_rrf", verdict, conf)]
            score_res = compute_truthfulness_score(rs, hybrid_direct_evidence=de)
            label     = score_res["label"]
            score     = score_res["score"]
            rule      = score_res["rule_applied"]

            ok = label == case["expected_label"]
            if ok:
                passed += 1

            print(f"   verdict         = {verdict}")
            print(f"   confidence      = {conf:.4f}")
            print(f"   direct_evidence = {de}")
            print(f"   score           = {score}")
            print(f"   label           = {label}")
            print(f"   rule            = {rule}")
            print(f"   expected        = {case['expected_label']}")
            print(f"   ---> {'PASS' if ok else 'FAIL'}")

            # Print claim-level details
            for i, d in enumerate(result.get("details", [])[:2]):
                print(f"   [claim {i}] verdict={d.get('verdict')} reasoning={str(d.get('reasoning',''))[:80]}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"   ERROR: {exc}")

    print("\n" + "=" * 65)
    print(f"RESULT: {passed}/{len(CASES)} cases passed")
    print("=" * 65)


asyncio.run(main())
