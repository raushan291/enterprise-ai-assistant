def build_rag_prompt(query: str, context_docs: list) -> str:
    """
    Build a RAG prompt using retrieved context and user query.

    Args:
        query (str): The user's question.
        context_docs (list): A list of dicts, each with keys like {"text": ..., "metadata": ...}.

    Returns:
        str: The full prompt to send to the language model.
    """

    # Define your instruction
    system_instruction = (
        "You are a knowledgeable and precise assistant. "
        "Use the provided context to answer the user's question truthfully. "
        "If the answer cannot be found in the context, say you don't know."
    )

    # Add context
    context_text = "\n---\n".join(
        [doc.get("text", "") for doc in context_docs if doc.get("text")]
    )

    # Final prompt
    prompt = (
        f"{system_instruction}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    return prompt
