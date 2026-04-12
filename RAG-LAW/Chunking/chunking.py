import json
import os

def chunk_text(text, chunk_size_words, overlap_words):
    """
    Hàm chia nhỏ văn bản thành các chunk dựa trên số từ (words).
    Khoảng trượt (sliding window) theo số overlap để đảm bảo ngữ cảnh không bị đứt đoạn.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return []
        
    for i in range(0, len(words), chunk_size_words - overlap_words):
        chunk = " ".join(words[i:i + chunk_size_words])
        if chunk:
            chunks.append(chunk)
            
    return chunks

def process_chunking(input_path, output_path, chunk_size_words=200, overlap_words=40):
    if not os.path.exists(input_path):
        print(f"File không tồn tại: {input_path}")
        return
        
    print(f"Đang đọc dữ liệu từ: {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    all_chunks = []
    
    for item in data:
        law_title = item.get("law_title", "").strip()
        article = item.get("article", "").strip()
        url = item.get("url", "")
        content = item.get("content", "").strip()
        
        if not content:
            continue
            
        # Dọn dẹp lỗi khoảng trắng dư trước dấu câu (vd: "Điều 2 ." -> "Điều 2.") xảy ra do split/join hoặc do crawl
        import re
        article = re.sub(r'\s+([.,:;!?])', r'\1', article).strip()
        content = re.sub(r'\s+([.,:;!?])', r'\1', content).strip()
        law_title = re.sub(r'\s+([.,:;!?])', r'\1', law_title).strip()
            
        # Nối Metadata vào đầu mỗi chunk để Embedding (VectorDB) hiểu đúng ngữ cảnh pháp lý
        context_prefix = f"Văn bản: {law_title} | Điều khoản: {article} - Nội dung: "
        
        words_in_content = content.split()
        
        # Nếu điều trích xuất ngắn hơn giới hạn chunk, giữ nguyên toàn bộ không cắt
        if len(words_in_content) <= chunk_size_words:
            all_chunks.append({
                "law_title": law_title,
                "article": article,
                "url": url,
                "chunk_id": 1,
                "text": context_prefix + content
            })
            continue
            
        # Chia nhỏ nếu vượt quá kích thước
        chunks = chunk_text(content, chunk_size_words, overlap_words)
        for i, chunk in enumerate(chunks):
            # Với mỗi chunk, ta vẫn gán lại context để không bao giờ bị lạc mất thông tin bộ luật
            all_chunks.append({
                "law_title": law_title,
                "article": article,
                "url": url,
                "chunk_id": i + 1,
                "text": context_prefix + chunk
            })
            
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
    print(f"Đã tạo thành công {len(all_chunks)} chunks từ {len(data)} bài/điều luật nguyên bản.")
    print(f"Lưu tại: {output_path}")

if __name__ == "__main__":
    # Đường dẫn file input (Dữ liệu đã dọn dẹp sạch \n bên clean.py)
    input_file = r"D:\Fake-news-detections\RAG-LAW\Data\law_articles_cleaned.json"
    
    # Đường dẫn file output dành cho nhúng Vector DB
    output_file = r"D:\Fake-news-detections\RAG-LAW\Data\law_chunks.json"
    
    # Cấu hình Chunking: Phù hợp cho model Embedding đa ngôn ngữ (vd: PhoBERT, bge-m3)
    # 150 từ tiếng Việt ~ khoảng 200 - 250 tokens
    CHUNK_SIZE_WORDS = 500
    OVERLAP_WORDS = 100
    
    process_chunking(input_file, output_file, CHUNK_SIZE_WORDS, OVERLAP_WORDS)
