"""
utils/cache.py
Disk-based caching using diskcache.
Reduces redundant LLM calls and embedding computations.
"""

import hashlib
import json
from typing import Any, Optional, Callable
from functools import wraps

import diskcache
from config.settings import settings
from utils.logger import logger


class CacheManager:
    """
    Persistent disk cache for expensive operations:
    - LLM responses
    - Embeddings
    - Cypher query results
    """

    def __init__(self):
        self.cache = diskcache.Cache(
            directory=settings.cache.cache_dir,
            timeout=1,
        )
        self.ttl = settings.cache.ttl_seconds
        logger.info(f"Cache initialized at: {settings.cache.cache_dir}")

    def _make_key(self, prefix: str, data: Any) -> str:
        """Create a deterministic cache key."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        hash_val = hashlib.md5(serialized.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve from cache."""
        value = self.cache.get(key)
        if value is not None:
            logger.debug(f"Cache HIT: {key[:40]}...")
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store in cache."""
        expire = ttl if ttl is not None else self.ttl
        self.cache.set(key, value, expire=expire)
        logger.debug(f"Cache SET: {key[:40]}...")

    def get_or_compute(
        self, prefix: str, data: Any, compute_fn: Callable, ttl: Optional[int] = None
    ) -> Any:
        """Get from cache or compute and store."""
        key = self._make_key(prefix, data)
        cached = self.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self.set(key, result, ttl=ttl)
        return result

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        self.cache.delete(key)

    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        logger.warning("Cache cleared!")

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self.cache),
            "directory": settings.cache.cache_dir,
            "ttl_seconds": self.ttl,
        }


# Singleton
cache_manager = CacheManager()


def cached(prefix: str, ttl: Optional[int] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key_data = {"args": str(args), "kwargs": str(kwargs)}
            return cache_manager.get_or_compute(
                prefix=f"{prefix}:{func.__name__}",
                data=key_data,
                compute_fn=lambda: func(*args, **kwargs),
                ttl=ttl,
            )
        return wrapper
    return decorator


__all__ = ["cache_manager", "cached", "CacheManager"]
