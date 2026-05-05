import numpy as np
import urllib.request
import json
from typing import List, Union

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBED_MODEL      = "bge-m3:latest"
EMBED_DIM        = 1024


def _ollama_embed(texts: List[str]) -> np.ndarray:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    vecs = np.array(data["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def get_embedder():
    return _OllamaEmbedder()


class _OllamaEmbedder:

    def encode(
        self,
        sentences,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]
        result = _ollama_embed(sentences)
        if len(sentences) == 1:
            return result[0]
        return result


class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = None):
        self.dim = EMBED_DIM

    def embed_documents(self, texts: Union[str, List[str]], batch_size: int = 16) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.strip() for t in texts if t.strip()]
        return _ollama_embed(texts)

    def embed_query(self, query: str) -> np.ndarray:
        return _ollama_embed([query])[0]
