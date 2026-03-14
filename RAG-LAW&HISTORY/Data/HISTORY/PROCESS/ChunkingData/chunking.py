import pandas as pd

path = r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\history.csv"

df = pd.read_csv(path)

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_lenght = len(text)
    
    while start < text_lenght:
        end = min(start + chunk_size, text_lenght)
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

df["content_en"] = df["content_en"].apply(chunk_text)

df = df.explode("content_en").reset_index(drop=True)

df["content_en"] = df["title_en"] + ": " + df["content_en"]

df.to_csv(r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\history_chunked.csv", index=False, encoding='utf-8-sig')  