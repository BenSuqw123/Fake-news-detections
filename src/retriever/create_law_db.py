import sys
import json
import os
import gc
from pathlib import Path
import chromadb
from tqdm import tqdm
import torch
from src.config import PROJECT_ROOT, MODELS_DIR

def build_law_database_resumable(json_path: str = None):
    if json_path is None:
        json_path = PROJECT_ROOT / "RAG_LAW/Data/law_chunks.json"
    else:
        json_path = Path(json_path)
        
    db_path = MODELS_DIR / "law_chroma"
    if not json_path.exists():
        print(f"Không tìm thấy file: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Tổng số chunks trong file: {len(chunks)}")

    torch.set_num_threads(4)
    print("Đang khởi tạo model BGE-M3 (Device: CPU)...")
    embedder = BGEM3Embedder(device='cpu')
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name="law",
        metadata={"hnsw:space": "cosine"}
    )

    existing_ids = set()
    if collection.count() > 0:
        existing_ids = set(collection.get(include=[])["ids"])

    print(f"Số lượng vector đã có trong DB: {len(existing_ids)}")

  
    to_process = []
    for item in chunks:
        doc_id = str(item["id"]) 

        if doc_id not in existing_ids:
            to_process.append(item)

    print(f"Cần thêm mới: {len(to_process)}")

    if not to_process:
        print("Database đã cập nhật đầy đủ!")
        return

   
    BATCH_SIZE = 8  
    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Indexing"):
        batch = to_process[i:i + BATCH_SIZE]
        batch_ids = [str(x["id"]) for x in batch]
        batch_texts = [x["text"] for x in batch] 
        batch_metadatas = [
            {
                "law_title": x.get("law_title", "Không rõ"),
                "article": x.get("article", "Không rõ"),
                "url": x.get("url", ""),
                "chunk_id": str(x["id"])
            }
            for x in batch
        ]

        try:
            embeddings = embedder.embed_documents(batch_texts)

            collection.add(
                ids=batch_ids,
                embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )

        except Exception as e:
            print(f"\nLỗi tại batch {i}: {e}")
            continue

        if i % (BATCH_SIZE * 5) == 0:
            gc.collect()

    print(f"\nHOÀN THÀNH. Tổng số vector trong DB: {collection.count()}")


if __name__ == "__main__":
    build_law_database_resumable()