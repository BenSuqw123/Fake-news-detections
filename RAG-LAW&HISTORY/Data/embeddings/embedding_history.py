from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

df = pd.read_csv(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\history_chunked.csv')
texts = df['content_en'].fillna("").astype(str).tolist()

instruction = "Represent this sentence for searching relevant passages: "
processed_texts = [instruction + t for t in texts]

embeddings = model.encode(processed_texts, 
                          batch_size=32, 
                          show_progress_bar=True,
                          convert_to_numpy=True
)

df['bge_vector'] = list(embeddings)
df.to_pickle(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\embeddings\history_bge_embedded.pkl')

print("Hoàn thành!")