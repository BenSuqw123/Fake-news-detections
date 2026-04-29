import chromadb

CHROMA_CLIENT = None
EMBEDDER = None

def load_chroma():
    global CHROMA_CLIENT, EMBEDDER

    if CHROMA_CLIENT is None:
        db_path = r"C:\Users\User\Desktop\Fake news detection\Fake-news-detections\RAG_LAW\Models\law_chroma"
        CHROMA_CLIENT = chromadb.PersistentClient(
            path=str(db_path)
        )

    if EMBEDDER is None:
        from src.retriever.embedder import BGEM3Embedder
        EMBEDDER = BGEM3Embedder(device='cpu')

    collection = CHROMA_CLIENT.get_collection("law")
    return collection, EMBEDDER


def search_chroma(query: str, top_k=3):
    try:
        collection, embedder = load_chroma()

        query_vec = embedder.embed_documents([query])[0]

        results = collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k
        )

        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": results["distances"][0][i],
                "method": "vector"
            })

        return docs

    except Exception as e:
        print(f"Chroma error: {e}")
        return []