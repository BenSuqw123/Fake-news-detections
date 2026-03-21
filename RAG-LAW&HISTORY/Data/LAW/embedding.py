import os
import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI(api_key="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")

csv_path = r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\LAW\courtlistener_chunked.csv"
df = pd.read_csv(csv_path)

df["content"] = df["content"].fillna("").astype(str)
documents = df["content"].tolist()

def embed_texts(texts, model="text-embedding-3-large", batch_size=100): 
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype="float32")

embeddings = embed_texts(documents)

print("Embedding shape:", embeddings.shape)