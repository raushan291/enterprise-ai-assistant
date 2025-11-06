from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.rag_pipeline.orchestrator import rag_response

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


@router.post("/", response_model=RAGResponse)
def query(inp: QueryIn) -> RAGResponse:
    """Handle a user query by running the Retrieval Augmented Generation (RAG) pipeline."""
    return rag_response(inp.query)
