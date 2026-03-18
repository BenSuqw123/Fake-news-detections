import pandas as pd
path=r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_6000_clean.csv"
# df = pd.read_parquet(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_6000_clean.parquet')
# df.to_csv(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_6000_clean.csv', index=False)

df = pd.read_csv(path)
df = df.drop(columns=['date_filed','source'])
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_lenght = len(text)
    
    while start < text_lenght:
        end = min(start + chunk_size, text_lenght)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

df["content"] = df["content"].apply(chunk_text)

df = df.explode("content").reset_index(drop=True)

df["content"] = df["title"] + ": " + df["content"]

df.to_csv(r"D:\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked.csv", index=False, encoding='utf-8-sig')  
