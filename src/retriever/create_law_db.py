import sys
import json
import os
import gc
from pathlib import Path
import chromadb
from tqdm import tqdm
import torch

# Thiết lập project_root
project_root = Path("/content/drive/MyDrive/Fake-news-detections")
os.chdir(project_root)

# Thêm đường dẫn vào sys.path
sys.path.insert(0, str(project_root))

try:
    from src.retriever.embedder import BGEM3Embedder
except ImportError:
    print("❌ Lỗi: Kiểm tra lại cấu trúc thư mục src/retriever/embedder.py")
    sys.exit(1)

def build_law_database_resumable_v2():
    print("🚀 Đang khởi động tiến trình Embedding tối ưu trên CPU...")

    json_path = project_root / "RAG-LAW/Data/law_articles_cleaned.json"
    db_path = str(project_root / "RAG-LAW/Models/law_chroma")

    # 1. Đọc dữ liệu
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
    print(f"📂 Đã tải: {len(articles)} điều luật.")

    # 2. Khởi tạo Embedder (CPU)
    # Tối ưu số luồng CPU để không bị nghẽn
    torch.set_num_threads(4) 
    embedder = BGEM3Embedder(device='cpu')
    
    # 3. Khởi tạo ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="law",
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Kiểm tra Resume
    existing_ids = set()
    if collection.count() > 0:
        # Lấy ID theo từng đợt để tránh treo RAM nếu DB quá lớn
        existing_ids = set(collection.get(include=[])['ids'])
    
    print(f"📊 Đã có {len(existing_ids)} bản ghi trong DB. Đang lọc dữ liệu mới...")

    # Chuẩn bị danh sách cần xử lý
    to_process = []
    for i, item in enumerate(articles):
        # Dùng ID dựa trên nội dung hoặc index cố định để Resume chính xác
        doc_id = f"law_{i}" 
        if doc_id not in existing_ids:
            to_process.append((doc_id, item))

    if not to_process:
        print("✅ Dữ liệu đã đầy đủ. Không cần chạy thêm.")
        return

    print(f"🔥 Cần xử lý thêm: {len(to_process)} điều luật.")

    # 5. Vòng lặp xử lý tối ưu
    # Trên CPU Colab, batch_size từ 4-8 là điểm cân bằng giữa tốc độ và RAM
    OPTIMAL_BATCH = 8 
    
    for i in tqdm(range(0, len(to_process), OPTIMAL_BATCH), desc="Indexing"):
        batch = to_process[i : i + OPTIMAL_BATCH]
        
        batch_ids = [x[0] for x in batch]
        # Giới hạn 2500 ký tự để CPU xử lý nhanh và vẫn đủ ngữ cảnh luật
        batch_texts = [x[1].get("content", "")[:2500] for x in batch]
        batch_metadatas = [{
            "title": x[1].get("law_title", "N/A"),
            "article": x[1].get("article", "N/A"),
            "url": x[1].get("url", "N/A")
        } for x in batch]

        try:
            # Embedding
            embeddings = embedder.embed_documents(batch_texts)
            
            # Lưu vào DB
            collection.add(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            # Dọn dẹp RAM sau mỗi 10 batch
            if i % (OPTIMAL_BATCH * 10) == 0:
                gc.collect()
                
        except Exception as e:
            print(f"\n⚠️ Lỗi tại batch {i}: {e}. Đang bỏ qua để tiếp tục...")
            continue

    print(f"\n✅ Hoàn thành! Tổng số bản ghi hiện tại: {collection.count()}")

if __name__ == "__main__":
    build_law_database_resumable_v2()