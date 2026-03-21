"""Build FAISS index using local BGE."""

import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR, DATA_DIR
from src.retriever.embedder import BGEEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_clean_chunks(data_dir: Path) -> pd.DataFrame:
    law_path = data_dir / "LAW" / "courtlistener_chunked.csv"
    hist_path = data_dir / "HISTORY" / "RAW" / "history_chunked.csv"
    
    logger.info("Loading LAW...")
    df_law = pd.read_csv(law_path)
    df_law['text'] = df_law['content'].fillna('').astype(str)
    df_law['title'] = df_law.get('title', '').fillna('')
    df_law['url'] = df_law.get('url', '').fillna('')
    df_law['source'] = 'law'
    
    logger.info("Loading HISTORY...")
    df_hist = pd.read_csv(hist_path)
    df_hist['text'] = df_hist['content_en'].fillna('').astype(str)
    df_hist['title'] = df_hist['title_en'].fillna('')
    df_hist['url'] = df_hist.get('url', '').fillna('')
    df_hist['source'] = 'history'
    
    df = pd.concat([df_law[['text', 'title', 'url', 'source']], df_hist[['text', 'title', 'url', 'source']]], ignore_index=True)
    
    orig_n = len(df)
    df = df[df['text'].str.len() > 20].drop_duplicates(subset=['text']).reset_index(drop=True)
    logger.info(f"Cleaned: {orig_n} → {len(df)} chunks")
    return df

def build_and_save_index(df: pd.DataFrame):
    MODELS_DIR.mkdir(exist_ok=True)
    
    texts = df['text'].tolist()
    embedder = BGEEmbedder()
    embeddings = embedder.embed_documents(texts)
    
    faiss.normalize_L2(embeddings)
    dim = 384  # BGE-small
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    index_path = MODELS_DIR / "knowledge.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS: {index_path} ({index.ntotal} vecs, dim {dim})")
    
    metadata_path = MODELS_DIR / "metadata.jsonl"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Metadata"):
            f.write(json.dumps({'text': row['text'], 'title': row['title'], 'url': row['url'], 'source': row['source']}) + '\n')
    logger.info(f"Saved metadata: {metadata_path}")

if __name__ == "__main__":
    df = load_clean_chunks(DATA_DIR)
    build_and_save_index(df)
    print("Index built - run python src/main.py")

