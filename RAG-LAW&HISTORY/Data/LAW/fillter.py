import pandas as pd
import os

# 1. Cấu hình đường dẫn (Đảm bảo file này tồn tại trên ổ D)
source_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked.csv"
output_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked_final.csv"

# Kiểm tra file trước khi đọc
if not os.path.exists(source_path):
    print(f"❌ Không tìm thấy file tại: {source_path}")
else:
    # 2. Đọc file
    df = pd.read_csv(source_path)
    
    # 3. CẮT DỮ LIỆU XUỐNG 3000 DÒNG ĐẦU TIÊN
    df = df.head(2000)
    print(f"✅ Đã lấy {len(df)} dòng dữ liệu gốc.")

    # 4. Loại bỏ cột không cần thiết
    cols_to_drop = ['date_filed', 'source']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 5. Hàm chia nhỏ văn bản (Chunking)
    def chunk_text(text, chunk_size=500, overlap=100):
        if not isinstance(text, str):
            return []
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    # 6. Xử lý Chunking
    print("⏳ Đang thực hiện chunking...")
    df["content"] = df["content"].apply(lambda x: chunk_text(x, chunk_size=500, overlap=100))

    # Bung các list thành các dòng riêng biệt
    df = df.explode("content").reset_index(drop=True)

    # Kết hợp Title vào Content để tăng ngữ cảnh cho RAG
    df["content"] = df["title"].astype(str) + ": " + df["content"].astype(str)

    # 7. Lưu file mới
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"🚀 Hoàn thành! File mới có {len(df)} dòng (sau khi chunking) đã được lưu tại: {output_path}")