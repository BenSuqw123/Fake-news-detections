import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from api.router import router
from src.config import LLM_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
   
    import asyncio
    loop = asyncio.get_event_loop()

    print("[Startup] Pre-warming underthesea word_tokenize (main thread) …")
    from underthesea import word_tokenize as _wt
    _wt("khởi động", format="text")   # single warm-up call; model now loaded
    print("[Startup] underthesea ready.")

    print("[Startup] Pre-warming BM25 index …")
    from src.retriever.search_bm25 import load_bm25
    await loop.run_in_executor(None, load_bm25)
    print("[Startup] BM25 ready.")

    print("[Startup] Pre-warming ChromaDB …")
    from src.retriever.search_chromadb import load_chroma
    await loop.run_in_executor(None, load_chroma)
    print("[Startup] ChromaDB ready.")

    print("[Startup] Pre-warming embedder …")
    from src.retriever.embedder import get_embedder
    await loop.run_in_executor(None, get_embedder)
    print("[Startup] Embedder ready.")

    print("[Startup] ✓ All models ready — accepting requests.")
    yield


app = FastAPI(title="Vietnamese Fake News Detection System", lifespan=lifespan)

app.include_router(router)

static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    index_file = os.path.join(static_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "index.html not found in static folder."}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<rect width="100" height="100" rx="20" fill="#2563EB"/>'
        '<path d="M50 12 L80 22 L80 52 C80 70 66 82 50 88 '
        "C34 82 20 70 20 52 L20 22 Z\" fill='white' opacity='0.92'/>"
        '<path d="M35 51 L45 62 L65 40" stroke="#2563EB" stroke-width="7" '
        'fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model":  LLM_MODEL,
        "chroma": "connected",
        "bm25":   "loaded",
    }
