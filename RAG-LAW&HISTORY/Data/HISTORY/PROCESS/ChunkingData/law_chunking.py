import pandas as pd

path = r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_6000_clean.csv"

df = pd.read_csv(path)
df = df.drop(columns=['date_filed', 'source'], errors='ignore')

df["content"] = df["content"].fillna("").astype(str)
df["title"] = df["title"].fillna("").astype(str)

def chunk_by_paragraph(text, max_chars=1200, overlap_chars=200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # overlap bằng phần cuối chunk trước
            overlap_text = current_chunk[-overlap_chars:] if current_chunk else ""
            current_chunk = (overlap_text + " " + para).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

df["chunks"] = df["content"].apply(chunk_by_paragraph)
df = df.explode("chunks").dropna(subset=["chunks"]).reset_index(drop=True)

df["content"] = df["title"] + ": " + df["chunks"]
df = df.drop(columns=["chunks"])

df.to_csv(
    r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked.csv",
    index=False,
    encoding="utf-8-sig"
)