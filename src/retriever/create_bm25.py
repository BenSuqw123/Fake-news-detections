import re
import json
import os
import pickle
import time
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize


def tokenize(text: str) -> list[str]:
    """
    Hybrid tokenizer combining underthesea compounds + individual syllables.

    underthesea produces compound tokens like 'đảng_cộng_sản_việt_nam' which
    are precise but too specific — a query containing just 'đảng' won't match.
    Adding individual syllables (from a clean whitespace split) means both
    compound and partial-word queries hit the same document.

    Punctuation is stripped from the syllable pass so "năm," → "năm".
    Deduplication preserves order: compound tokens come first (higher BM25
    specificity), syllables fill in gaps.
    """
    compound_tokens = word_tokenize(text, format="text").lower().split()
    clean_text      = re.sub(r"[^\w\s]", " ", text.lower())
    syllable_tokens = clean_text.split()

    seen   = set()
    result = []
    for t in compound_tokens + syllable_tokens:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


def build_bm25_standalone():
    # Resolve paths relative to this file so the script works on any machine
    project_root   = Path(__file__).resolve().parent.parent.parent
    json_path      = project_root / "RAG_LAW" / "Data" / "law_chunks.json"
    bm25_save_path = project_root / "RAG_LAW" / "Models" / "bm25" / "bm25_database.pkl"

    if not json_path.exists():
        print(f"Không tìm thấy file: {json_path}")
        return

    print(f"Đang đọc dữ liệu từ {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Tổng số chunks: {len(chunks)}")
    print("Đang tách từ (hybrid: compound + syllable) cho các chunks...")

    t0 = time.time()
    tokenized_corpus = [
        tokenize(item.get("text", ""))
        for item in tqdm(chunks, desc="Tokenizing")
    ]
    tokenize_secs = time.time() - t0
    print(f"Tokenization done in {tokenize_secs:.1f}s")

    print("Đang huấn luyện mô hình BM25...")
    t1  = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    train_secs = time.time() - t1
    print(f"BM25 training done in {train_secs:.1f}s")

    data_to_save = {
        "bm25_model": bm25,
        "metadata":   chunks,
    }

    os.makedirs(bm25_save_path.parent, exist_ok=True)
    with open(bm25_save_path, "wb") as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = bm25_save_path.stat().st_size / 1024 / 1024
    total_secs = tokenize_secs + train_secs
    print(f"Index saved → {bm25_save_path}")
    print(f"Index size:  {size_mb:.2f} MB")
    print(f"Total time:  {total_secs:.1f}s")


if __name__ == "__main__":
    build_bm25_standalone()
