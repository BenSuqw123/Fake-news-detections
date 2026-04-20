import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

class BGEM3Embedder:
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading BGE-M3 on {self.device}...")
        
        self.model = SentenceTransformer(model_name, device=self.device)
        
        self.dim = 1024 
        
        print(f"BGE-M3 loaded successfully. Dimension: {self.dim}")

    def embed_documents(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        texts = [t.strip() for t in texts if t.strip()]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
