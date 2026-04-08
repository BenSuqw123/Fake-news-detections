import sys
import json
from pathlib import Path
import chromadb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR
from src.retriever.embedder import BGEEmbedder

def build_law_database():
    json_path = Path("d:/Fake-news-detections/RAG-LAW/Data/law_articles_cleaned.json")
    
    if not json_path.exists():
        print(f"Không tìm thấy file dữ liệu: {json_path}")
        return
        
    print("Đang tải dữ liệu JSON...")
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    print(f"Đã tải: {len(articles)} điều luật.")
    
    print("Khởi tạo BGE Embedder...")
    embedder = BGEEmbedder()
    
    db_path = str(MODELS_DIR / "law")
    print(f"Khởi tạo ChromaDB tại: {db_path}")
    
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        client.delete_collection(name="law")
        print("Đã xóa collection cũ.")
    except Exception:
        pass
        
    collection = client.create_collection(name="law")
    
    batch_size = 500
    for i in range(0, len(articles), batch_size):
        batch = articles[i : i + batch_size]
        
        texts_to_embed = [item["content"] for item in batch]
        ids = [f"law_{i+j}" for j in range(len(batch))]
        metadatas = [{
            "title": item["law_title"],
            "article_name": item["article"],
            "url": item["url"],
            "source": "law"
        } for item in batch]
        
        print(f"Đang Embedding batch {i//batch_size + 1}/{(len(articles)//batch_size)+1}...")
        embeddings = embedder.embed_documents(texts_to_embed)
        
        print(f"Đang lưu vào ChromaDB...")
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts_to_embed,
            metadatas=metadatas
        )

    print(f"Đã lưu toàn bộ vector vào ChromaDB ('law' domain).")

if __name__ == "__main__":
    build_law_database()
