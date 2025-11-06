from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthStatus(BaseModel):
    """Schema for the health check response."""

    status: str


@router.get("/", response_model=HealthStatus)
def health() -> HealthStatus:
    """Health check endpoint to verify the API is running."""
    return HealthStatus(status="ok")
