from typing import Any, TypedDict

from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage
from prometheus_client import Histogram

from src.models.inference import generate_response
from src.utils.cache import cache_get, cache_set
from src.utils.conversation_memory import ConversationMemory

from .prompt_template import build_rag_prompt
from .retriever import retrieve_context

# Prometheus metrics
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", "Time to retrieve context"
)
INFERENCE_LATENCY = Histogram(
    "rag_inference_latency_seconds", "Time to generate response"
)

# Conversation memory setup
memory = ConversationMemory(max_turns=10)

# Guardrails safety setup
guard = Guard().use_many(
    ToxicLanguage(
        threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION
    ),
    DetectPII(
        pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.EXCEPTION
    ),
)


class RAGResponse(TypedDict):
    query: str
    context: list[Any]
    answer: str
    cached: bool


def rag_response(query: str) -> RAGResponse:
    """Generate a RAG (Retrieval-Augmented Generation) response for a given user query."""
    # Pre-safety check - before retrieval
    try:
        guard.validate(query)
    except Exception as exc:
        return {
            "query": query,
            "context": [],
            "answer": f"The input contains unsafe or sensitive content and cannot be processed: {str(exc)}",
            "cached": False,
        }

    conversation_context = memory.get_context()
    contextual_query = (
        f"Previous conversation:\n{conversation_context}\n\nCurrent query:\n{query}"
    )

    # Check cache first
    cached_data = cache_get("eka_response", query)

    if cached_data:
        return {
            "query": query,
            "context": cached_data["context"],
            "answer": cached_data["answer"],
            "cached": True,
        }

    # Run normal RAG
    with RETRIEVAL_LATENCY.time():
        context = retrieve_context(contextual_query, top_k=3)
    base_prompt = build_rag_prompt(query, context)

    # Inject conversation context *before* the final question-answer block
    if conversation_context.strip():
        full_prompt = (
            f"Previous conversation:\n{conversation_context}\n\n"
            f"Current query:\n{base_prompt}"
        )
    else:
        full_prompt = base_prompt

    with INFERENCE_LATENCY.time():
        answer = generate_response(full_prompt)

    # Post-safety check - after generation
    try:
        guard.validate(answer)
    except Exception:
        answer = "The generated response contained unsafe or sensitive content and was blocked."

    # Update conversation memory
    memory.add_turn(query, answer)

    # Store result in cache
    cache_set("eka_response", query, {"context": context, "answer": answer})

    return {
        "query": query,
        "context": context,
        "answer": answer,
        "cached": False,
    }
