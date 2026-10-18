"""
Cache Utilities
Provides simple file-based caching for news articles to reduce API calls.
"""

import json
import os
import time
from pathlib import Path

# Cache directory relative to project root
_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Default cache TTL: 15 minutes
DEFAULT_TTL_SECONDS = 15 * 60


def _cache_path(cache_key: str) -> Path:
    """Return the file path for a given cache key."""
    safe_key = "".join(c if c.isalnum() else "_" for c in cache_key)
    return _CACHE_DIR / f"{safe_key}.json"


def get_cached(cache_key: str, ttl: int = DEFAULT_TTL_SECONDS) -> list | None:
    """
    Retrieve cached data if it exists and is not expired.

    Args:
        cache_key: Unique identifier for this cache entry
        ttl: Time-to-live in seconds

    Returns:
        Cached data or None if cache miss/expired
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if time.time() - cached["timestamp"] > ttl:
            path.unlink(missing_ok=True)  # Delete expired cache
            return None

        return cached["data"]
    except Exception:
        return None


def set_cache(cache_key: str, data: list) -> None:
    """
    Save data to cache.

    Args:
        cache_key: Unique identifier for this cache entry
        data: Data to cache (must be JSON-serializable)
    """
    path = _cache_path(cache_key)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": time.time(), "data": data}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[Cache] Failed to write cache: {e}")


def clear_cache() -> int:
    """Clear all cached files. Returns number of files deleted."""
    count = 0
    for f in _CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            count += 1
        except Exception:
            pass
    return count


def cache_stats() -> dict:
    """Return stats about the current cache."""
    files = list(_CACHE_DIR.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_size_kb": round(total_size / 1024, 2),
    }
