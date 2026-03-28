import json
import numpy as np
import chromadb
import faiss
import os
import shutil
from tqdm import tqdm

FAISS_PATH = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Models\knowledge.faiss"
JSONL_PATH = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Models\metadata.jsonl"
DB_PATH = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\VectorBD\chroma_db"

def load_embeddings_from_faiss(path):
    """Đọc vector 384 chiều từ file FAISS."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file FAISS: {path}")
    index = faiss.read_index(path)
    logger_msg = f"Đang load {index.ntotal} vectors, Dimension: {index.d}"
    print(logger_msg)
    return index.reconstruct_n(0, index.ntotal).astype('float32')

def import_to_chroma():
    if os.path.exists(DB_PATH):
        print(f"Đang xóa Database cũ tại {DB_PATH}...")
        shutil.rmtree(DB_PATH)

    client = chromadb.PersistentClient(path=DB_PATH)
    collection_name = "fake_news_rag"
    collection = client.create_collection(name=collection_name)
    print(f"✨ Đã tạo Collection mới: {collection_name}")

    embeddings_np = load_embeddings_from_faiss(FAISS_PATH)
    
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        all_metadata = [json.loads(line) for line in f]
    
    total_records = len(all_metadata)
    print(f"Tổng số bản ghi cần nạp: {total_records}")

    batch_size = 2000
    for i in tqdm(range(0, total_records, batch_size), desc="Đang nạp vào ChromaDB"):
        end_idx = min(i + batch_size, total_records)
        
        batch_ids = [f"doc_{j}" for j in range(i, end_idx)]
        batch_vectors = embeddings_np[i:end_idx].tolist()
        subset_meta = all_metadata[i:end_idx]
        
        batch_docs = [item.get('text', '') for item in subset_meta]
        
        batch_metas = [{
            "topic": item.get('source', 'unknown'),
            "title": str(item.get('title', 'N/A'))[:500],
            "url": str(item.get('url', 'N/A'))
        } for item in subset_meta]

        collection.add(
            ids=batch_ids,
            embeddings=batch_vectors,
            metadatas=batch_metas,
            documents=batch_docs
        )

    print(f"\n THÀNH CÔNG! Đã nạp {collection.count()} bản ghi vào VectorDB.")
    return collection

if __name__ == "__main__":
    try:
        final_coll = import_to_chroma()
        
        print("\n--- KIỂM TRA DỮ LIỆU ---")
        test_res = final_coll.peek(1)
        print(f"Nội dung mẫu: {test_res['documents'][0][:150]}...")
        print(f"Metadata mẫu: {test_res['metadatas'][0]}")
        
    except Exception as e:
        print(f" Lỗi: {e}")