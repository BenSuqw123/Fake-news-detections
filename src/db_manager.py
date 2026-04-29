import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.retriever.create_law_db import build_law_database_resumable
from src.retriever.create_bm25 import build_bm25_standalone

def rebuild_all_indexes(json_chunks_path: str):
    """Rebuilds both ChromaDB and BM25 indexes synchronously to prevent drift."""
    print(f"Rebuilding indexes from {json_chunks_path}")
    print("--- Rebuilding ChromaDB ---")
    build_law_database_resumable(json_chunks_path)
    
    print("--- Rebuilding BM25 ---")
    build_bm25_standalone(json_chunks_path)
    print("[OK] Both indexes rebuilt in sync.")
