from fastapi import FastAPI
from src.api.routes.query import router as query_router
from src.api.routes.health import router as health_router

app = FastAPI(title="EKA - Enterprise Knowledge Assistant")
app.include_router(health_router, prefix="/health")
app.include_router(query_router, prefix="/query")

# uvicorn src.api.main:app --reload