from fastapi import APIRouter
from pydantic import BaseModel
from src.rag_pipeline.orchestrator import rag_response

router = APIRouter()

class QueryIn(BaseModel):
    query: str

@router.post("/")
def query(inp: QueryIn):
    return rag_response(inp.query)