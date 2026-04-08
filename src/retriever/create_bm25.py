import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer

def build_bm25_standalone():
    project_root = Path(r"D:\Fake-news-detections")
    json_path = project_root / "RAG-LAW/Data/law_articles_cleaned.json"
    bm25_save_path = project_root / "RAG-LAW/Models/bm25/bm25_model.pkl"

    print(f"Đang đọc dữ liệu từ {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print("Đang tách từ tiếng Việt (Tokenizing)...")
    corpus = [item.get("content", "") for item in articles]
    
    tokenized_corpus = [
        ViTokenizer.tokenize(doc).lower().split() 
        for doc in tqdm(corpus, desc="Processing")
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(bm25_save_path.parent, exist_ok=True)
    with open(bm25_save_path, 'wb') as f:
        pickle.dump(bm25, f)

    print(f"File BM25 đã nằm tại: {bm25_save_path}")

if __name__ == "__main__":
    build_bm25_standalone()