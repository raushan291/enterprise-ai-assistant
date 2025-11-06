import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer model for generating text embeddings."""
    model = SentenceTransformer(model_name)
    return model


def embed_texts(embedder: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings using a SentenceTransformer model."""
    embeddings = embedder.encode(texts, show_progress_bar=True)
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()
    elif hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy().tolist()
    return embeddings
