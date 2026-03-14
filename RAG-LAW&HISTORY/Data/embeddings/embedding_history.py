import pandas as pd

df = pd.read_json(
    r"C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\wikimedia_5000_clean.jsonl",
    lines=True
)

print(df.head())