import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from user_input_processing.claim_extractor import extract_atomic_claims
from user_input_processing.check_input import hybrid_search, async_extract_key_info, async_verify_claim, filter_relevant_docs

article = """Hiến pháp năm 2013 khẳng định Việt Nam là một quốc gia độc lập và có chủ quyền. Điều 1 của Hiến pháp quy định Việt Nam là nước độc lập, thống nhất và toàn vẹn lãnh thổ. Đồng thời, Hiến pháp cũng nêu rõ Nhà nước Việt Nam là nhà nước pháp quyền xã hội chủ nghĩa của nhân dân, do nhân dân, vì nhân dân. Tuy nhiên, Điều 4 lại khẳng định Đảng Cộng sản Việt Nam là lực lượng lãnh đạo Nhà nước và xã hội. Ngoài ra, một số người hiểu sai rằng Hiến pháp 2013 cho phép các tổ chức chính trị khác có quyền lãnh đạo tương đương, điều này là không đúng với nội dung pháp luật."""

async def run_eval():
    print("## 1. CLAIMS EXTRACTION")
    claims = await extract_atomic_claims(article)
    for i, c in enumerate(claims):
        print(f"Claim {i+1}: {c['claim']} (Verifiable: {c['is_verifiable']})")
    
    for i, c in enumerate(claims):
        if not c.get('is_verifiable'): continue
        print(f"\n## 2 & 3. TEST FOR Claim {i+1}: {c['claim']}")
        claim_text = c['claim']
        try:
            clean_query = await async_extract_key_info(claim_text)
            final_query = f"{claim_text}, {clean_query}"
            all_docs = await asyncio.to_thread(hybrid_search, final_query, 3)
            # Apply filter to match actual pipeline condition
            filtered_docs = filter_relevant_docs(claim_text, all_docs)
            
            print("[RETRIEVAL]")
            for j, doc in enumerate(all_docs):
                 print(f"Doc {j+1} [{doc['method']}]: {doc['text'][:100]}... | Norm Score: {doc.get('normalized_score')}")
                 
            print("\n[VERIFICATION]")
            res = await async_verify_claim(claim_text, filtered_docs)
            print(f"VERDICT: {res['verdict']}")
            print(f"CONFIDENCE: {res['confidence']}")
            print(f"EVIDENCE: {res['evidence']}")
            print(f"REASON: {res['reason']}")
            
        except Exception as e:
             print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_eval())
