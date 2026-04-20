import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from underthesea import word_tokenize 

def build_bm25_standalone():
    project_root = Path(r"D:\Fake-news-detections")
    json_path = project_root / "RAG-LAW/Data/law_chunks.json" 
    bm25_save_path = project_root / "RAG-LAW/Models/bm25/bm25_database.pkl"

    if not json_path.exists():
        print(f"Không tìm thấy file: {json_path}")
        return

    print(f"Đang đọc dữ liệu từ {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Tổng số chunks: {len(chunks)}")

    print("Đang tách từ tiếng Việt (Underthesea) cho các chunks...")
    corpus = [item.get("text", "") for item in chunks] 
    
    tokenized_corpus = [
        word_tokenize(doc, format="text").lower().split() 
        for doc in tqdm(corpus, desc="Tokenizing")
    ]

    print("Đang huấn luyện mô hình BM25")
    bm25 = BM25Okapi(tokenized_corpus)

    data_to_save = {
        "bm25_model": bm25,
        "metadata": chunks  
    }

    os.makedirs(bm25_save_path.parent, exist_ok=True)
    
    with open(bm25_save_path, 'wb') as f:
        pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Hệ thống BM25 đã sẵn sàng tại: {bm25_save_path}")

if __name__ == "__main__":
    build_bm25_standalone()