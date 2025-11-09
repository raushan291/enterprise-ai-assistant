from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel

from src.rag_pipeline.orchestrator import memory, rag_response
from src.utils.cache import cache_clear

router = APIRouter()


class QueryIn(BaseModel):
    """Input schema for a RAG query."""

    query: str


class RAGResponse(BaseModel):
    """Response schema for a RAG result."""

    query: str
    context: list[Any]
    answer: str
    cached: bool


class ResetResponse(BaseModel):
    """Response schema for a reset operation result."""

    status: str
    message: str


class ChatTurn(BaseModel):
    """Response schema for history result."""

    role: Literal["user", "assistant"]
    text: str


@router.post("/", response_model=RAGResponse)
def query(inp: QueryIn) -> RAGResponse:
    """Handle a user query by running the Retrieval Augmented Generation (RAG) pipeline."""
    return rag_response(inp.query)


@router.post("/reset", response_model=ResetResponse)
def reset_conversation() -> ResetResponse:
    """Reset conversation state by clearing in-memory chat history and cached RAG outputs."""
    memory.clear()  # clear conversation turns
    cache_clear("eka_response")  # clear cached RAG outputs
    return ResetResponse(status="success", message="Conversation reset successfully.")


@router.get("/history", response_model=list[ChatTurn])
def get_history():
    """Return the stored conversation history"""
    history = memory.get_history()
    return history
