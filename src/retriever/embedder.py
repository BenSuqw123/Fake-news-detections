"""Local BGE Embedder (sentence-transformers, 384 dim, 100% offline/free)."""

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.config import MODELS_DIR

class BGEEmbedder:
    """BAAI/bge-small-en-v1.5 embedder for RAG."""
    
    def __init__(self):
        self.model_path = MODELS_DIR / "bge"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder=str(self.model_path))
        self.dim = 384
        print(f"BGE loaded, dim: {self.dim}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        texts = [t.strip() for t in texts if t.strip()]
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        return embeddings

if __name__ == "__main__":
    embedder = BGEEmbedder()
    test = embedder.embed_documents(["Test law", "Test history"])
    print(f"Test shape: {test.shape}")

