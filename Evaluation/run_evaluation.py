import asyncio
import json
import time
import sys
import io
import os
import argparse
from colorama import Fore, Style, init
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class Logger(object):
    def __init__(self, filename="evaluation_log.txt", mode="w"):
        self.terminal = sys.stdout
        self.log = open(filename, mode, encoding="utf-8")
        if mode == "a":
            self.log.write("\n" + "="*60 + "\n")
            self.log.write(f"--- CONTINUED AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.log.write("="*60 + "\n\n")
        else:
            self.log.write("="*60 + "\n")
            self.log.write(f"--- NEW EVALUATION RUN AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            self.log.write("="*60 + "\n\n")

    def write(self, message):
        self.terminal.write(message)

        import re
        clean_message = re.sub(r'\x1b\[[0-9;]*m', '', message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

init(autoreset=True)

sys.path.insert(0, '.')

print("[Startup] Pre-warming underthesea word_tokenize …")
from underthesea import word_tokenize as _wt
_wt("khởi động", format="text")
print("[Startup] underthesea ready.")

from src.pipeline import run_pipeline

DATASET_FILE = "Evaluation/evaluation_dataset.json"
REPORT_FILE = "Evaluation/evaluation_report.json"
DEFAULT_START_ID = os.getenv("EVAL_START_ID")
DEFAULT_START_INDEX = int(os.getenv("EVAL_START_INDEX", "1"))

def load_resume_results(report_file=REPORT_FILE):

    if not os.path.exists(report_file):
        return [], set()

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
        details = report.get("details", [])
        completed_ids = {
            r.get("id")
            for r in details
            if r.get("id") and r.get("ai_verdict") not in (None, "CRASH")
        }
        return details, completed_ids
    except Exception as e:
        print(f"{Fore.YELLOW}[WARN] Cannot load old report for resume: {e}")
        return [], set()

def save_report(results, dataset, total_time):

    result_by_id = {r.get("id"): r for r in results if r.get("id")}
    ordered_results = [
        result_by_id[item.get("id")]
        for item in dataset
        if item.get("id") in result_by_id
    ]

    correct_count = sum(1 for r in ordered_results if r.get("is_correct"))
    accuracy = (correct_count / len(dataset)) * 100 if dataset else 0

    category_stats = {}
    for r in ordered_results:
        cat = r.get("category", "Unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if r.get("is_correct"):
            category_stats[cat]["correct"] += 1

    report = {
        "summary": {
            "total_cases": len(dataset),
            "completed_cases": len(ordered_results),
            "correct": correct_count,
            "accuracy": round(accuracy, 2),
            "total_time_seconds": round(total_time, 2)
        },
        "category_breakdown": category_stats,
        "details": ordered_results
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return report

def first_detail(pipeline_out):
    details = pipeline_out.get("details") or []
    return details[0] if details else {}

def split_partial_eval_claims(query):
    """
    Splits a query into multiple claims for PARTIAL evaluation cases.
    Supports splitting by ' và ', ' nhưng ', ' mặc dù ', ' tuy nhiên ', and semicolons.
    """
    text = (query or "").strip()
    suffix = text[-1] if text.endswith((".", "?", "!")) else "."
    core = text.rstrip(".?! ")

    # List of separators to try splitting with
    separators = [" và ", " nhưng ", " mặc dù ", " tuy nhiên ", ";"]
    
    parts = []
    used_sep = None
    for sep in separators:
        if sep in core:
            parts = [p.strip(" ,;") for p in core.split(sep) if p.strip(" ,;")]
            if len(parts) >= 2:
                used_sep = sep
                break
    
    if not parts:
        return None

    prefix = partial_subject_prefix(parts[0])
    claims = [parts[0]]
    
    for part in parts[1:]:
        # Later clauses in PARTIAL cases often omit the subject after "và".
        starts_with_subject = prefix and part.lower().startswith(prefix.lower())
        looks_dependent = part[:1].islower() or part.lower().startswith((
            "bị ", "được ", "phải ", "có ", "không ", "nhận ", "là ",
            "làm ", "nộp ", "lập ", "chờ ", "đổi ", "trả ", "thu ",
        ))

        if prefix and not starts_with_subject and looks_dependent:
            claims.append(f"{prefix} {part}")
        else:
            claims.append(part)

    return [
        {
            "claim": c + suffix,
            "keywords": c,
            "entities": [],
            "type": "FACT",
            "is_verifiable": True,
        }
        for c in claims
        if len(c) >= 10
    ]

def partial_subject_prefix(first_clause):
    markers = (
        " \u0111\u01b0\u1ee3c ",
        " b\u1ecb ",
        " ph\u1ea3i ",
        " c\u00f3 ",
        " kh\u00f4ng ",
        " ngh\u1ec9 ",
        " nh\u1eadn ",
        " l\u00e0m ",
        " n\u1ed9p ",
    )
    lower = first_clause.lower()
    positions = [lower.find(m) for m in markers if lower.find(m) > 0]
    if positions:
        return first_clause[:min(positions)].strip()

    words = first_clause.split()
    return " ".join(words[:4]) if len(words) >= 2 else ""

def print_header():
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}HỆ THỐNG ĐÁNH GIÁ TỰ ĐỘNG - LAW RAG PIPELINE")
    print(f"{Fore.CYAN}{'='*60}")
    print(f"{Fore.WHITE}Dataset: {Fore.YELLOW}{DATASET_FILE}")
    print(f"{Fore.WHITE}Report:  {Fore.YELLOW}{REPORT_FILE}")
    print(f"{Fore.WHITE}Log File: {Fore.YELLOW}evaluation_log.txt")
    print(f"{Fore.CYAN}{'-'*60}\n")

async def process_dataset(resume=False, start_id=None, start_index=None):
    if not os.path.exists(DATASET_FILE):
        print(f"{Fore.RED}Lỗi: Không tìm thấy file dữ liệu {DATASET_FILE}")
        return

    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Lỗi đọc file JSON: {e}")
        return

    print_header()
    print(f"{Fore.BLUE}[INFO] Đã nạp {len(dataset)} câu hỏi. Đang khởi tạo luồng xử lý...\n")

    should_load_old = not args.reset and os.path.exists(REPORT_FILE)
    
    if should_load_old:
        results, completed_ids = load_resume_results(REPORT_FILE)
    else:
        results, completed_ids = [], set()

    correct_count = sum(1 for r in results if r.get("is_correct"))
    start_time = time.time()

    if resume and completed_ids:
        print(f"{Fore.YELLOW}[RESUME] Found {len(completed_ids)} completed cases in {REPORT_FILE}.")
        print(f"{Fore.YELLOW}[RESUME] Skipping completed IDs and continuing remaining cases.\n")

    start_pos = 0
    if start_index is not None:
        start_pos = max(0, start_index - 1)
    if start_id:
        matched = next(
            (i for i, item in enumerate(dataset) if item.get("id") == start_id),
            None,
        )
        if matched is None:
            print(f"{Fore.RED}[ERROR] start_id not found in dataset: {start_id}")
            return
        start_pos = matched

    if start_pos > 0:
        print(f"{Fore.YELLOW}[START] Running from #{start_pos + 1}/{len(dataset)} ID: {dataset[start_pos].get('id')}\n")

    for i, item in enumerate(dataset[start_pos:], start=start_pos):
        idx = i + 1
        total = len(dataset)
        item_id = item.get("id", "N/A")

        should_skip = (item_id in completed_ids) and not getattr(args, 'overwrite', False)
        
        if should_skip:
            print(f"{Fore.YELLOW}[{idx}/{total}] Skip completed ID: {item_id}")
            print(f"{Style.DIM}{'-' * 45}")
            continue

        print(f"{Fore.WHITE}[{idx}/{total}] Xử lý ID: {Fore.BLUE}{item_id}")
        print(f"  Query: {item['query'][:80]}...")

        case_start = time.time()
        try:

            pre_claims = None
            if item.get("expected_verdict") == "PARTIAL":
                pre_claims = split_partial_eval_claims(item["query"])
                if pre_claims and len(pre_claims) > 1:
                    print(f"    Eval split: {len(pre_claims)} claims")

            pipeline_out = await run_pipeline(
                item["query"],
                retrieval_method="hybrid_rrf",
                pre_extracted_claims=pre_claims,
                mode="eval",
            )
            detail = first_detail(pipeline_out)

            ai_verdict = pipeline_out.get("verdict", "ERROR")
            expected = item["expected_verdict"]
            is_correct = (ai_verdict == expected)
            case_duration = time.time() - case_start

            if is_correct:
                correct_count += 1
                status_icon = f"{Fore.GREEN}ĐÚNG"
            else:
                status_icon = f"{Fore.RED}SAI"

            print(f"    Kết quả: {status_icon} {Fore.WHITE}| AI: {Fore.MAGENTA}{ai_verdict:<12}")
            print(f"    Thời gian: {Fore.CYAN}{case_duration:.2f}s")
            if not is_correct:
                reason = detail.get('reason') or detail.get('reasoning') or pipeline_out.get("reason") or 'N/A'
                print(f"    Reason: {reason}")
                print(f"    Evidence: {detail.get('evidence') or 'N/A'}")
                print(f"    Contra evidence: {detail.get('contradicting_evidence') or 'N/A'}")

            results.append({
                "id": item.get("id"),
                "query": item["query"],
                "category": item.get("category", "General"),
                "expected_verdict": expected,
                "ai_verdict": ai_verdict,
                "is_correct": is_correct,
                "confidence": pipeline_out.get("confidence", 0.0),
                "reasoning": detail.get("reason") or detail.get("reasoning") or pipeline_out.get("reason", ""),
                "evidence": detail.get("evidence", "N/A"),
                "direct_evidence": detail.get("direct_evidence"),
                "contradicting_evidence": detail.get("contradicting_evidence"),
                "eval_metrics": pipeline_out.get("eval_metrics", {}),
                "processing_time_ms": pipeline_out.get("processing_time_ms", 0)
            })

        except Exception as e:
            print(f"    {Fore.RED}LỖI: {str(e)}")
            results.append({"id": item.get("id"), "ai_verdict": "CRASH", "is_correct": False, "error": str(e)})

        save_report(results, dataset, time.time() - start_time)
        print(f"{Style.DIM}{'-' * 45}")

    total_time = time.time() - start_time
    report = save_report(results, dataset, total_time)
    accuracy = report["summary"]["accuracy"]
    correct_count = report["summary"]["correct"]

    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.GREEN}HOÀN TẤT ĐÁNH GIÁ")
    print(f"{Fore.WHITE}Tổng cộng:   {Fore.YELLOW}{len(dataset)} câu")
    print(f"{Fore.WHITE}Chính xác:   {Fore.GREEN}{correct_count} ({round(accuracy, 2)}%)")
    print(f"{Fore.WHITE}Thời gian:   {Fore.CYAN}{round(total_time, 2)}s")
    print(f"{Fore.WHITE}Báo cáo JSON: {Fore.MAGENTA}{REPORT_FILE}")
    print(f"{Fore.WHITE}Log File TXT: {Fore.MAGENTA}evaluation_log.txt")
    print(f"{Fore.CYAN}{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation dataset.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip IDs already present in evaluation_report.json (Default if file exists).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing report and start from scratch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing results in evaluation_report.json.",
    )
    parser.add_argument(
        "--start-id",
        default=DEFAULT_START_ID,
        help="Start running from a dataset ID, for example C-01.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=DEFAULT_START_INDEX,
        help="Start running from a 1-based dataset index, for example 21.",
    )
    args = parser.parse_args()

    append_log = args.resume or args.start_id or (args.start_index and args.start_index > 1)
    logger = Logger("evaluation_log.txt", mode="a" if append_log else "w")
    sys.stdout = logger
    sys.stderr = logger
    if args.resume or args.start_id or args.start_index:
        print(f"\n{Fore.YELLOW}{'='*60}")
        print(f"{Fore.YELLOW}CONTINUE EVALUATION")
        print(f"{Fore.YELLOW}{'='*60}\n")

    if args.reset and os.path.exists(REPORT_FILE):
        print(f"{Fore.RED}[RESET] Deleting existing report {REPORT_FILE}...")
        try:
            os.remove(REPORT_FILE)
        except:
            pass

    asyncio.run(
        process_dataset(
            resume=args.resume or not args.reset,
            start_id=args.start_id,
            start_index=args.start_index,
        )
    )
