import pandas as pd
import os

source_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked.csv"
output_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked_final.csv"

if not os.path.exists(source_path):
    print(f"Không tìm thấy file tại: {source_path}")
else:
    df = pd.read_csv(source_path)
    
    df = df.head(2000)
    print(f"Đã lấy {len(df)} dòng dữ liệu gốc.")

    cols_to_drop = ['date_filed', 'source']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

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

    df["content"] = df["content"].apply(lambda x: chunk_text(x, chunk_size=500, overlap=100))

    df = df.explode("content").reset_index(drop=True)

    df["content"] = df["title"].astype(str) + ": " + df["content"].astype(str)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Hoàn thành! File mới có {len(df)} dòng (sau khi chunking) đã được lưu tại: {output_path}")