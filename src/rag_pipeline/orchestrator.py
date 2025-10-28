from .retriever import retrieve_context
from src.models.inference import generate_response

def rag_response(query: str):
    context = retrieve_context(query, top_k=3)
    # simple prompt
    prompt = """Use the context to answer the question.\n\nContext:\n"""
    for c in context:
        prompt += c.get("text", "") + "\n---\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    answer = generate_response(prompt)
    return {"query": query, "context": context, "answer": answer}