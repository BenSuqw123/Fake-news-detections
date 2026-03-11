import pandas as pd

# Đọc file JSONL
df = pd.read_json("D:\Fake-news-detections\RAG\LAW\courtlistener_6000_clean.jsonl", lines=True)

# Lưu sang Parquet (cần cài đặt: pip install pyarrow)
df.to_parquet("D:\Fake-news-detections\RAG\LAW\courtlistener_6000_clean.parquet", index=False)