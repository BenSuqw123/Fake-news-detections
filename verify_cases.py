# test_both_retrievers.py
import sys
sys.path.insert(0, '.')

# Pre-warm underthesea trước khi import bất kỳ thứ gì
from underthesea import word_tokenize as _wt
_wt('khoi dong', format='text')

import urllib.request
import json
import chromadb
from pathlib import Path
from src.retriever.search_bm25 import search_bm25, _EXPANSION_MAP

query = "Theo Hiến pháp 2013, công dân Việt Nam có quyền tự do thành lập đảng phái chính trị độc lập"

print("=" * 60)
print("QUERY:", query)
print("=" * 60)

# ==================== BM25 ====================
print("\n[BM25] Testing via search_bm25()...")
print("[BM25] Expansion map:", list(_EXPANSION_MAP.keys()))

results_bm25 = search_bm25(query, top_k=5)
print(f"[BM25] Top {len(results_bm25)} docs:")
for i, r in enumerate(results_bm25):
    print(f"  #{i+1} score={r['score']:.3f} | {r['text'][:120]}")

dieu4_bm25 = any('điều 4' in r['text'].lower() for r in results_bm25)
print(f"[BM25] Điều 4 trong top 5: {'✓ CÓ' if dieu4_bm25 else '✗ KHÔNG'}")

# ==================== ChromaDB ====================
print("\n[ChromaDB] Testing...")

payload = json.dumps({"model": "bge-m3:latest", "input": [query]}).encode()
req = urllib.request.Request(
    "http://localhost:11434/api/embed",
    data=payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    vec = json.loads(resp.read())["embeddings"][0]

print(f"[ChromaDB] Embedding OK — dim={len(vec)}")

db = chromadb.PersistentClient(
    path=Path(r'C:\Users\ACER\Documents\Desktop\Fake News Detection\Fake-news-detections\RAG_LAW\Models\law_chroma').as_posix()
)
col = db.get_collection("law")
results_chroma = col.query(query_embeddings=[vec], n_results=5)

print("[ChromaDB] Top 5 docs:")
for i, (doc, dist) in enumerate(zip(results_chroma["documents"][0], results_chroma["distances"][0])):
    sim = 1 - dist / 2
    print(f"  #{i+1} sim={sim:.3f} | {doc[:120]}")

dieu4_chroma = any('điều 4' in doc.lower() for doc in results_chroma["documents"][0])
print(f"[ChromaDB] Điều 4 trong top 5: {'✓ CÓ' if dieu4_chroma else '✗ KHÔNG'}")

# ==================== SUMMARY ====================
print("\n" + "=" * 60)
print("SUMMARY")
print(f"  BM25:     Điều 4 {'✓ TÌM THẤY' if dieu4_bm25 else '✗ KHÔNG TÌM THẤY'}")
print(f"  ChromaDB: Điều 4 {'✓ TÌM THẤY' if dieu4_chroma else '✗ KHÔNG TÌM THẤY'}")
if not dieu4_bm25 and not dieu4_chroma:
    print("  → Cả 2 đều miss → LLM không có context → reasoning sai")
elif not dieu4_bm25:
    print("  → BM25 miss → expansion map chưa đủ token")
elif not dieu4_chroma:
    print("  → ChromaDB miss → embedding không capture được context")
else:
    print("  → Cả 2 đều tìm thấy Điều 4 ✓ — vấn đề nằm ở evaluation prompt")
print("=" * 60)