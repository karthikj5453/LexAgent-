from openai import AsyncOpenAI
from config import settings
import os
import logging

logger = logging.getLogger("lexagent.embedder")

embed_client = None

def get_embed_client() -> AsyncOpenAI:
    global embed_client
    if embed_client is None:
        api_key = settings.nim_api_key
        if not api_key:
            api_key = os.environ.get("NIM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy_key"
        embed_client = AsyncOpenAI(
            base_url=settings.nim_base_url,
            api_key=api_key,
        )
    return embed_client

async def embed_texts(texts: list[str], input_type: str = "passage") -> list[list[float]]:
    """Batch embed texts using NIM's embedding model."""
    if not texts:
        return []
    
    client = get_embed_client()
    try:
        extra_body = {}
        if "nv-embedqa" in settings.nim_embedding_model.lower():
            extra_body = {"input_type": input_type}
            
        response = await client.embeddings.create(
            model=settings.nim_embedding_model,
            input=texts,
            encoding_format="float",
            extra_body=extra_body,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error embedding texts: {e}")
        # Return mock embeddings (zeros) in test mode if API key is not configured/valid
        if settings.nim_api_key == "" or settings.nim_api_key == "dummy_key":
            logger.warning("Mocking zero embeddings for testing.")
            return [[0.0] * 1024 for _ in texts]
        raise e

async def embed_query(query: str) -> list[float]:
    results = await embed_texts([query], input_type="query")
    return results[0]
