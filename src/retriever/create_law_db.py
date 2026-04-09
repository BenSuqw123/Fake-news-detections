import sys
import json
import os
import gc
from pathlib import Path
import chromadb
from tqdm import tqdm
import torch

project_root = Path(r"D:\Fake-news-detections")
os.chdir(project_root)

sys.path.insert(0, str(project_root))

try:
    from src.retriever.embedder import BGEM3Embedder
except ImportError:
    print("Lỗi: Kiểm tra lại cấu trúc thư mục src/retriever/embedder.py")
    sys.exit(1)

def build_law_database_resumable():
    json_path = project_root / "RAG-LAW/Data/law_articles_cleaned.json"
    db_path = str(project_root / "RAG-LAW/Models/law_chroma")

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"Đã tải: {len(articles)} điều luật.")

    torch.set_num_threads(4) 
    embedder = BGEM3Embedder(device='cpu')
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="law",
        metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])['ids'])
    
    print(f"Đã có {len(existing_ids)} bản ghi trong DB. Đang lọc dữ liệu mới...")

    to_process = []
    for i, item in enumerate(articles):
        doc_id = f"law_{i}" 
        if doc_id not in existing_ids:
            to_process.append((doc_id, item))

    if not to_process:
        print("Dữ liệu đã đầy đủ.")
        return

    print(f"Cần xử lý thêm: {len(to_process)} điều luật.")

    OPTIMAL_BATCH = 8 
    
    for i in tqdm(range(0, len(to_process), OPTIMAL_BATCH), desc="Indexing"):
        batch = to_process[i : i + OPTIMAL_BATCH]
        
        batch_ids = [x[0] for x in batch]
        batch_texts = [x[1].get("content", "")[:2500] for x in batch]
        batch_metadatas = [{
            "title": x[1].get("law_title", "N/A"),
            "article": x[1].get("article", "N/A"),
            "url": x[1].get("url", "N/A")
        } for x in batch]

        try:
            embeddings = embedder.embed_documents(batch_texts)
            
            collection.add(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            if i % (OPTIMAL_BATCH * 10) == 0:
                gc.collect()
                
        except Exception as e:
            print(f"\nLỗi tại batch {i}: {e}. Đang bỏ qua để tiếp tục...")
            continue

    print(f"\nTổng số bản ghi hiện tại: {collection.count()}")

if __name__ == "__main__":
    build_law_database_resumable()