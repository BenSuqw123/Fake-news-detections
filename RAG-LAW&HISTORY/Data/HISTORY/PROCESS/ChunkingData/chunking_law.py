import pandas as pd
import os

# 1. Define Paths (Separate Source and Destination)
source_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked_3000.csv"
output_path = r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked_final.csv"

# 2. Load Data
if not os.path.exists(source_path):
    print(f"Error: Could not find {source_path}")
else:
    df = pd.read_csv(source_path)
    
    # Drop unnecessary columns safely
    cols_to_drop = ['date_filed', 'source']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 3. Improved Chunking Function
    def chunk_text(text, chunk_size=500, overlap=100):
        if not isinstance(text, str):
            return []
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunks.append(text[start:end])
            # Move the window forward by (size - overlap)
            start += (chunk_size - overlap)
            
            # Safety break if overlap >= chunk_size
            if chunk_size <= overlap: break 
            
        return chunks

    # 4. Process Data
    print("Chunking in progress...")
    df["content"] = df["content"].apply(lambda x: chunk_text(x, chunk_size=500, overlap=100))

    # Explode the list of chunks into individual rows
    df = df.explode("content").reset_index(drop=True)

    # Combine Title and Content for better RAG context
    df["content"] = df["title"].astype(str) + ": " + df["content"].astype(str)

    # 5. Save Results
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Done! Saved {len(df)} chunks to {output_path}")