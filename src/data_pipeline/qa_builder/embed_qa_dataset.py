import json
import os

import numpy as np

from src.config.settings import settings
from src.data_pipeline.embedder import embed_texts, get_embedder


def embed_qa_dataset(
    qa_file: str, output_dir: str, source_file: str
) -> list[list[float]]:
    """Embed generated QA pairs and save embeddings + metadata."""
    os.makedirs(output_dir, exist_ok=True)
    embedder = get_embedder(settings.EMBEDDING_MODEL)

    with open(qa_file, "r") as f:
        qa_pairs = json.load(f)

    texts = [item["answer"] for item in qa_pairs]
    embeddings = embed_texts(embedder, texts)

    metadata = []
    for idx, pair in enumerate(qa_pairs):
        metadata.append(
            {
                "id": idx,
                "question": pair["question"],
                "answer": pair["answer"],
                "source": os.path.basename(source_file),
                "chunk_id": f"chunk_{idx:04d}",
                "type": "qa",
            }
        )

    np.save(os.path.join(output_dir, "qa_embeddings.npy"), np.array(embeddings))
    with open(os.path.join(output_dir, "qa_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved QA embeddings + metadata to {output_dir}")
    return embeddings


def embed_chunks(
    chunks_file: str, output_dir: str, source_file: str
) -> list[list[str]]:
    """Embed raw text chunks (from preprocess step) and save embeddings + metadata."""
    os.makedirs(output_dir, exist_ok=True)
    embedder = get_embedder(settings.EMBEDDING_MODEL)

    # Read JSONL chunks file
    chunks = []
    with open(chunks_file, "r") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Assume each chunk entry has a 'text' field
    texts = [c.get("text", "") for c in chunks]
    embeddings = embed_texts(embedder, texts)

    metadata = []
    for idx, chunk in enumerate(chunks):
        metadata.append(
            {
                "id": idx,
                "source": os.path.basename(source_file),
                "chunk_id": chunk.get("id", f"chunk_{idx:04d}"),
                "type": "chunk",
                "meta": {
                    k: v for k, v in chunk.items() if k != "text"
                },  # keep other info if present
            }
        )

    np.save(os.path.join(output_dir, "chunk_embeddings.npy"), np.array(embeddings))
    with open(os.path.join(output_dir, "chunk_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved chunk embeddings + metadata to {output_dir}")
    return embeddings
