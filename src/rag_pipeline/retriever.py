from chromadb import Client
from sentence_transformers import SentenceTransformer

# placeholder: use in-process chroma
client = Client()

def retrieve_context(query: str, top_k: int = 3):
    # implement embedding + search
    # return list of dicts {"id":..., "text":..., "metadata":...}
    return []