import asyncio
import json
import sys
import time
from pathlib import Path

# Thêm đường dẫn gốc vào hệ thống
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from user_input_processing.Guardrail_input import check_input_validity
from user_input_processing.claimExtractor_getKeyword import extract_atomic_claims
from user_input_processing.rrf import hybrid_rrf_search
from user_input_processing.evaluation_layer import evaluate_final
from user_input_processing.chunking import split_text_into_chunks


async def async_process_retrieval(claim_text: str, keywords: str):
    """Tìm kiếm sử dụng keywords đã được trích xuất từ trước."""
    print(f"   [Search] Đang tìm bằng chứng cho: '{claim_text[:40]}...'")
    
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(None, hybrid_rrf_search, claim_text, keywords)
    
    return docs

async def process_single_claim(claim_data: dict, semaphore: asyncio.Semaphore):
    """Xử lý trọn gói 1 Claim: Dùng Keywords có sẵn -> Search -> Evaluation"""
    async with semaphore:
        claim_text = claim_data.get("claim", "")
        keywords = claim_data.get("keywords", "") # Lấy từ data đã trích xuất
        start_time = time.time()
        
        try:
            # Stage 3: Retrieval (Truyền thêm keywords vào)
            docs = await async_process_retrieval(claim_text, keywords)
            
            # Stage 4: Evaluation
            result = await evaluate_final(claim_text, docs)
            
        except Exception as e:
            result = {"verdict": "ERROR", "confidence": "0.0%", "reason": str(e)}

        result["claim"] = claim_text
        result["processing_time"] = f"{round(time.time() - start_time, 2)}s"
        return result

async def run_full_pipeline(user_input: str):
    start_total = time.time() 
    
    # 1. Guardrail Input
    print("\n[STAGE 1/3] Kiểm tra tính hợp lệ...")
    guardrail = await check_input_validity(user_input)
    if guardrail["status"] == "REJECT":
        return {"status": "REJECTED", "message": guardrail["message"]}

    clean_query = guardrail.get("clean_query", user_input)

    # 2. Claim Extraction (Gộp Claim + Keyword)
    print("[STAGE 2/3] Trích xuất các ý định và từ khóa...")
    input_chunks = split_text_into_chunks(clean_query, max_chars=1000)
    
    all_claims = []
    for chunk in input_chunks:
        # Gọi file claimExtractor_getKeyword đã tích hợp keywords
        results = await extract_atomic_claims(chunk) 
        all_claims.extend(results)
    
    if not all_claims:
        return {"status": "ERROR", "message": "Không tìm thấy ý định pháp lý."}

    # 3. Concurrent Processing
    print(f"[STAGE 3/3] Đang thẩm định dựa trên {len(all_claims)} nội dung...")
    semaphore = asyncio.Semaphore(1)
    
    # Tạo danh sách task xử lý song song
    tasks = [process_single_claim(c, semaphore) for c in all_claims]
    
    # Gather kết quả
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 4. Hậu xử lý kết quả
    clean_results = []
    for r in results:
        if isinstance(r, Exception):
            clean_results.append({"verdict": "ERROR", "reason": str(r), "claim": "Lỗi Task"})
        else:
            clean_results.append(r)

    # 5. Tổng hợp báo cáo
    final_report = {
        "overall_verdict": "CẦN KIỂM CHỨNG THÊM",
        "total_time": f"{round(time.time() - start_total, 2)}s",
        "summary": {
            "total": len(clean_results),
            "supported": sum(1 for r in clean_results if r.get("verdict") == "SUPPORTED"),
            "contradicted": sum(1 for r in clean_results if r.get("verdict") == "CONTRADICTED"),
            "insufficient": sum(1 for r in clean_results if r.get("verdict") == "INSUFFICIENT")
        },
        "details": clean_results
    }

    # Logic phán quyết tổng quát
    if final_report["summary"]["contradicted"] > 0:
        final_report["overall_verdict"] = "PHÁT HIỆN THÔNG TIN SAI LỆCH"
    elif final_report["summary"]["supported"] == len(clean_results) and len(clean_results) > 0:
        final_report["overall_verdict"] = "THÔNG TIN CHÍNH XÁC"

    return final_report

if __name__ == "__main__":
    test_input = "Hiến pháp 2013 quy định Việt Nam là quốc gia độc lập. Đồng thời cũng cho phép thiết lập các khu tự trị không tuân theo pháp luật chung."

    print("--- BẮT ĐẦU CHẠY HỆ THỐNG KIỂM CHỨNG ---")
    
    report = asyncio.run(run_full_pipeline(test_input))

    print("\n" + "="*30 + " BÁO CÁO JSON " + "="*30)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    
    print(report_json)
    print("="*74)