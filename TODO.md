# TODO.md - BGE Embeddings & KB Prep Progress

## Completed Steps
- [ ] Create src/retriever/__init__.py
- [ ] Create src/retriever/embedder.py
- [ ] Create src/retriever/prepare_knowledge.py
- [ ] Edit src/config.py (add EMBEDDING_MODEL_NAME, PROCESSED_DATA_PATH)
- [ ] Edit src/main.py (integrate BGEEmbedder in retriever)
- [ ] Run prepare_knowledge.py → generate knowledge_unified.pkl
- [ ] Venv install sentence_transformers if needed: pip install -r requirements.txt

## Notes
- BGE prefix for queries: "Represent this sentence for searching relevant passages: " (asymmetric search optimization).
- Model: bge-small-en-v1.5 (fast/local).
- Output: knowledge_unified.pkl w/ df['text', 'metadata', 'embedding'].

Update after each step.

