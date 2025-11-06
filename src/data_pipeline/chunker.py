from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_texts(
    texts: list[str], chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """Split a list of texts into smaller overlapping chunks for embedding or analysis."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks
