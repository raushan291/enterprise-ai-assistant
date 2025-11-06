import os

from chromadb import HttpClient

from src.config.settings import settings
from src.data_pipeline.qa_builder.embed_qa_dataset import embed_chunks, embed_qa_dataset
from src.data_pipeline.qa_builder.preprocess import preprocess_raw_data
from src.data_pipeline.qa_builder.qa_generator import generate_qa_pairs


def save_to_chroma(
    texts: list[str],
    embeddings: list[list[float]],
    source_name: str,
    source_type: str = "chunk",
) -> None:
    """Save text data and corresponding embeddings into a ChromaDB collection."""
    client = HttpClient(
        host=settings.CHROMA_SERVER_HOST, port=settings.CHROMA_SERVER_PORT
    )
    collection = client.get_or_create_collection(settings.TMP_COLLECTION_NAME)
    ids = [f"{source_name}_{source_type}_{i}" for i in range(len(embeddings))]
    metadatas = [
        {"source": source_name, "type": source_type} for _ in range(len(embeddings))
    ]

    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def run_pipeline() -> None:
    """Executes the full RAG dataset pipeline."""
    raw_files = [f for f in os.listdir(settings.DATA_RAW_DIR) if f.endswith(".txt")]

    for raw_file in raw_files:
        raw_path = os.path.join(settings.DATA_RAW_DIR, raw_file)
        base_name = os.path.splitext(raw_file)[0]

        chunks_file = f"{settings.DATA_PROCESSED_DIR}/{base_name}_chunks.jsonl"
        qa_file = f"{settings.DATA_QA_DIR}/{base_name}_qa_pairs.json"

        print(f"Processing {raw_file} ...")
        chunks = preprocess_raw_data(raw_path, chunks_file)
        # Embed and store raw chunks
        chunk_embeddings = embed_chunks(
            chunks_file, settings.DATA_PROCESSED_DIR, raw_path
        )
        save_to_chroma(
            settings.CHROMA_PATH,
            chunks,
            chunk_embeddings,
            base_name,
            source_type="chunk",
        )
        qa_pairs = generate_qa_pairs(chunks_file, qa_file)
        # Embed and store QA pairs
        qa_embeddings = embed_qa_dataset(qa_file, settings.DATA_PROCESSED_DIR, raw_path)
        save_to_chroma(
            settings.CHROMA_PATH, qa_pairs, qa_embeddings, base_name, source_type="qa"
        )

    print("RAG Dataset Pipeline completed for ALL raw files!")


if __name__ == "__main__":
    run_pipeline()
