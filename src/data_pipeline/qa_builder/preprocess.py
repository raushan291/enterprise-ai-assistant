import json
import os
import re

from docx import Document
from PyPDF2 import PdfReader


def load_raw_text(file_path: str) -> str:
    """Load and extract text from a file (.txt, .pdf, or .docx)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(
            [page.extract_text() for page in reader.pages if page.extract_text()]
        )

    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def clean_text(text: str) -> str:
    """Clean text by normalizing whitespace and removing excessive line breaks."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def preprocess_raw_data(input_path: str, output_path: str) -> list[str]:
    """Load, clean, and chunk raw text data, then save it as a JSONL file."""
    text = load_raw_text(input_path)

    text = clean_text(text)
    chunks = split_into_chunks(text)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for i, chunk in enumerate(chunks):
            json.dump({"id": i, "text": chunk}, f)
            f.write("\n")

    print(f"Saved {len(chunks)} chunks to {output_path}")
    return chunks
