import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from src.api.routes.health import router as health_router
from src.api.routes.query import router as query_router
from src.config.logging_config import get_logger
from src.config.settings import settings

# Configure basic logging
logger = get_logger(__name__)


# Lifespan event manager
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown events."""
    logger.info(f"Starting {settings.PROJECT_NAME} FastAPI service...")
    yield  # <-- App runs while inside this block
    logger.info(f"Shutting down {settings.PROJECT_NAME} service...")


# Initialize FastAPI app
app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

# Include routers
app.include_router(health_router, prefix="/health")
app.include_router(query_router, prefix="/query")


@app.get("/", tags=["Root"])
def root() -> dict:
    """Root endpoint to verify the API is running."""
    return {"message": "EKA up and running"}


# Prometheus metrics setup
REQUEST_COUNT = Counter("eka_request_count", "Total number of requests", ["endpoint"])
REQUEST_LATENCY = Histogram(
    "eka_request_latency_seconds", "Request latency (s)", ["endpoint"]
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next: callable) -> Response:
    """Middleware to collect Prometheus metrics for every HTTP request."""
    start_time = time.time()
    response = await call_next(request)
    latency = time.time() - start_time

    REQUEST_COUNT.labels(endpoint=request.url.path).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)

    return response


@app.get("/metrics", tags=["Monitoring"])
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
