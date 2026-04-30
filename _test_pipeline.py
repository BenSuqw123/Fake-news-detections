import asyncio, sys, time, json, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from src.pipeline import run_pipeline

async def test():
    query = "Theo Dieu 25 Hien phap 2013, cong dan co quyen tu do ngon luan va tu do bao chi."
    print("Query:", query)
    t0 = time.time()
    result = await run_pipeline(query, retrieval_method="hybrid_rrf")
    elapsed = time.time() - t0
    print("--- Pipeline result ---")
    print("verdict:        ", result.get("verdict"))
    print("confidence:     ", result.get("confidence"))
    de = result.get("direct_evidence", "")
    print("direct_evidence:", str(de).encode('ascii','replace').decode())
    print("time_ms:        ", int(elapsed * 1000))
    det = result.get("details", [{}])
    if det:
        r = str(det[0].get("reasoning", "")).encode('ascii','replace').decode()
        print("reasoning:      ", r[:300])

asyncio.run(test())
