"""Prepare unified knowledge base: load LAW/HISTORY chunks, clean, embed w/ BGE, save pkl."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add project root for imports

from src.config import DATA_DIR, PROCESSED_DATA_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chunks(data_dir: Path) -> pd.DataFrame:
    """Load LAW & HISTORY chunked CSVs."""
    law_path = data_dir / "LAW" / "courtlistener_chunked.csv"
    hist_path = data_dir / "HISTORY" / "RAW" / "history_chunked.csv"
    
    logger.info("Loading LAW chunks...")
    df_law = pd.read_csv(law_path)
    logger.info(f"LAW: {len(df_law)} rows, columns: {list(df_law.columns)}")
    
    logger.info("Loading HISTORY chunks...")
    df_hist = pd.read_csv(hist_path)
    logger.info(f"HISTORY: {len(df_hist)} rows, columns: {list(df_hist.columns)}")
    
    # Standardize text column (title,content,url for law; title_en,content_en,url for hist)
    df_law['text'] = df_law['content'].fillna('').astype(str)
    df_hist['text'] = df_hist['content_en'].fillna('').astype(str)
    
    # Metadata (keep title/url/source)
    df_law['source'] = 'law'
    df_hist['source'] = 'history'
    
    # Select common cols
    df_law_sel = df_law[['text', 'title', 'url', 'source']].copy()
    df_hist_sel = df_hist[['text', 'title_en', 'url', 'source']].rename(columns={'title_en': 'title'}).copy()
    
    df = pd.concat([df_law_sel, df_hist_sel], ignore_index=True)
    return df

def prepare_knowledge() -> Dict[str, Any]:
    """Main: clean, embed, save unified KB."""
    DATA_DIR.mkdir(exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(exist_ok=True)
    
    # Load
    df = load_chunks(DATA_DIR)
    
    # Clean: drop empty/dups
    orig_len = len(df)
    df = df[df['text'].str.len() > 10].drop_duplicates(subset=['text']).reset_index(drop=True)
    logger.info(f"Cleaned: {orig_len} -> {len(df)} rows")
    
    # Embed
    embedder = BGEEmbedder()
    embeddings = embedder.embed_documents(df['text'].tolist())
    df['embedding'] = list(embeddings)
    
    # Save pkl (preserves vectors)
    df[['text', 'embedding', 'title', 'url', 'source']].to_pickle(PROCESSED_DATA_PATH)
    logger.info(f"Saved knowledge_unified.pkl to {PROCESSED_DATA_PATH} ({len(df)} chunks)")
    
    return {'df': df, 'dim': embeddings.shape[1]}

if __name__ == "__main__":
    result = prepare_knowledge()
    print(f"Ready for FAISS: {len(result['df'])} docs, dim {result['dim']}")

