"""
Singleton async Groq client.
Import get_groq_client() everywhere instead of
instantiating groq.AsyncGroq() in each file.
"""
import groq
from src.config import GROQ_API_KEY

_client: groq.AsyncGroq | None = None

def get_groq_client() -> groq.AsyncGroq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        _client = groq.AsyncGroq(api_key=GROQ_API_KEY)
    return _client
