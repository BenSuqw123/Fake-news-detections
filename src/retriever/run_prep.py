"""Standalone runner for prepare_knowledge to fix import path issues."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Add CWD to path

from src.retriever.prepare_knowledge import prepare_knowledge

if __name__ == "__main__":
    prepare_knowledge()

