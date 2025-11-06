from chromadb import HttpClient
from chromadb.api.models.Collection import Collection

from src.config.settings import settings


def save_to_chroma(
    texts: list[str],
    embeddings: list[list[float]],
    source_name: str,
    source_type: str = "chunk",
) -> Collection:
    """Save a list of text documents and their embeddings into a ChromaDB collection."""
    client = HttpClient(
        host=settings.CHROMA_SERVER_HOST, port=settings.CHROMA_SERVER_PORT
    )
    collection = client.get_or_create_collection(settings.COLLECTION_NAME)

    ids = [f"{source_name}_{source_type}_{i}" for i in range(len(embeddings))]
    metadatas = [
        {"source": source_name, "type": source_type} for _ in range(len(embeddings))
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"Saved {len(texts)} chunks into Chroma DB")
    return collection
