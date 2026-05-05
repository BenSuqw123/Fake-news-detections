import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "RAG-LAW" / "Data"
TRANSFORMER_DIR: Path = PROJECT_ROOT / "Transformer"
RAW_DATA_DIR: Path = PROJECT_ROOT / "liar_dataset"
MODELS_DIR: Path = PROJECT_ROOT / "RAG-LAW" / "Models"

DATA_DIR = Path(os.getenv("DATA_DIR", DATA_DIR))
TRANSFORMER_DIR = Path(os.getenv("TRANSFORMER_DIR", TRANSFORMER_DIR))
MODELS_DIR = Path(os.getenv("MODELS_DIR", MODELS_DIR))

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-key-here")

# LLM - Groq API
MODEL_NAME: str       = "llama-3.3-70b-versatile"
LLM_MODEL: str        = MODEL_NAME
CLASSIFIER_MODEL: str = "llama-3.3-70b-versatile"
GROQ_API_KEY: str     = os.getenv("GROQ_API_KEY", "")

# Groq generation limits
MAX_TOKENS: int   = 256   
TEMPERATURE: float = 0.0   

# Embedding (multilingual, BAAI/bge-m3)
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
BGE_MODEL: str = EMBEDDING_MODEL_NAME
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "knowledge_unified.pkl"
CHUNK_SIZE: int = 1000

TOP_K_RETRIEVAL: int = 10  
TOP_K_RERANK: int    = 3    
DOC_CHAR_LIMIT: int  = 250 

EMBED_MODEL = "deprecated-openai"


def ensure_dirs() -> None:
    for dir_path in [DATA_DIR, TRANSFORMER_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Dir:     {DATA_DIR}")
    print(f"LLM Model:    {LLM_MODEL}")
    print(f"TOP_K_RETRIEVAL={TOP_K_RETRIEVAL}  TOP_K_RERANK={TOP_K_RERANK}")
    ensure_dirs()
    print("Directories ensured.")
