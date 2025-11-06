import os

import chromadb
import numpy as np
from chromadb import HttpClient
from chromadb.config import Settings

from src.config.settings import settings

# Chroma client
client = HttpClient(host=settings.CHROMA_SERVER_HOST, port=settings.CHROMA_SERVER_PORT)

# Embedding setup
try:
    from openai import OpenAI

    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    USE_OPENAI = bool(openai_client.api_key)
except Exception:
    USE_OPENAI = False

if not USE_OPENAI:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

# Load collection
collection = client.get_or_create_collection(name=settings.COLLECTION_NAME)


def _get_embedding(texts: str) -> list[list[float]]:
    """Return embeddings as list[list[float]] from OpenAI or MiniLM."""
    if USE_OPENAI:
        resp = openai_client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL, input=texts
        )
        return [item.embedding for item in resp.data]
    else:
        emb = embedder.encode(texts, convert_to_numpy=True)
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()
        elif hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy().tolist()
        return emb


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve top matching contexts for a given query.

    Args:
        query (str): user query
        top_k (int): number of most relevant chunks to return

    Returns:
        list[dict]: [{"id":..., "text":..., "metadata":...}]
    """
    # Compute embedding for the query
    query_emb = _get_embedding([query])[0]

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Format output
    contexts = []
    for idx, doc in enumerate(results.get("documents", [[]])[0]):
        contexts.append(
            {
                "id": results["ids"][0][idx],
                "text": doc,
                "metadata": (
                    results["metadatas"][0][idx] if results["metadatas"] else {}
                ),
                "score": (
                    1 - results["distances"][0][idx] if "distances" in results else None
                ),
            }
        )

    return contexts
