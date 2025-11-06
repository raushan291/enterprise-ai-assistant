import os

from src.config.settings import settings
from src.data_pipeline.chunker import chunk_texts
from src.data_pipeline.embedder import embed_texts, get_embedder
from src.data_pipeline.loader import load_data
from src.data_pipeline.vector_store import save_to_chroma

source_name = os.path.basename(settings.DATA_RAW_DIR)

# Load
texts = load_data(settings.DATA_RAW_DIR)
print(f"Loaded {len(texts)} raw documents.")

# Chunk
chunks = chunk_texts(texts, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
print(f"Created {len(chunks)} text chunks.")

# Embed
embedder = get_embedder(settings.EMBEDDING_MODEL)
embeddings = embed_texts(embedder, chunks)

# Save
save_to_chroma(chunks, embeddings, source_name, source_type="chunk")
