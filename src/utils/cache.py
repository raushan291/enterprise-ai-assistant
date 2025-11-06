import hashlib
import json
from typing import Any, Optional

import redis

from src.config.settings import settings


def get_redis_client() -> redis.Redis:
    """Create and return a Redis client instance using configured settings."""
    return redis.StrictRedis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        decode_responses=True,
    )


def _make_key(prefix: str, text: str) -> str:
    """Hash large text inputs for consistent short keys."""
    normalized = " ".join(
        text.strip().lower().split()
    )  # remove extra spaces, lowercase
    key_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    return f"{prefix}:{key_hash}"


def cache_get(prefix: str, text: str) -> Optional[dict[str, Any]]:
    """Get cached response for given query text."""
    client = get_redis_client()
    key = _make_key(prefix, text)
    val = client.get(key)
    if val:
        return json.loads(val)
    return None


def cache_set(prefix: str, text: str, value) -> None:
    """Cache value (Python dict, list, str) for given query."""
    client = get_redis_client()
    key = _make_key(prefix, text)
    client.setex(key, settings.CACHE_TTL, json.dumps(value))
