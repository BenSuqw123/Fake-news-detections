import pandas as pd

df1=pd.read_csv(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\wikimedia_5000_clean.csv')
df2=pd.read_csv(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\world_history_clean.csv')

df_merged = pd.concat([df1, df2], ignore_index=True)

df_merged = df_merged.drop_duplicates(subset=['url'])

df_merged.to_csv(r'D:\Fake-news-detections\RAG-LAW&HISTORY\Data\HISTORY\RAW\history.csv', index=False, encoding='utf-8-sig')

print(f"Tổng số dòng sau khi gộp và xóa trùng: {len(df_merged)}")
