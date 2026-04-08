"""Configuration module for Fake News Detection RAG System.

Defines dynamic paths relative to project root and environment variables.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file if exists
load_dotenv()

# Dynamically determine project root: parent of src/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "RAG-LAW" / "Data"
TRANSFORMER_DIR: Path = PROJECT_ROOT / "Transformer"
RAW_DATA_DIR: Path = PROJECT_ROOT / "liar_dataset"
MODELS_DIR: Path = PROJECT_ROOT / "RAG-LAW" / "Models"

# Environment overrides (optional)
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))
TRANSFORMER_DIR = Path(os.getenv("TRANSFORMER_DIR", TRANSFORMER_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", MODELS_DIR))

# API Keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-key-here")  # Set in .env

# Embedding constants (BGE local/free)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Fast, 384 dim; use bge-m3 for multi-lang
BGE_MODEL = EMBEDDING_MODEL_NAME  # Alias
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "knowledge_unified.pkl"
CHUNK_SIZE: int = 1000

# Backwards compat
EMBED_MODEL = "deprecated-openai"  # Use BGE now

def ensure_dirs() -> None:
    """Create necessary directories if missing."""
    for dir_path in [DATA_DIR, TRANSFORMER_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir: {DATA_DIR}")
    ensure_dirs()
    print("Directories ensured.")

