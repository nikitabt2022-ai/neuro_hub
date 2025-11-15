"""Embedding cache for reducing redundant computations."""

import hashlib
from typing import List, Optional
from collections import OrderedDict
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings."""

    def __init__(self, max_size: Optional[int] = None) -> None:
        """Initialize embedding cache.

        Args:
            max_size: Maximum cache size (None = use settings)
        """
        self.max_size = max_size or settings.embedding_cache_size
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Generate hash for text.

        Args:
            text: Input text

        Returns:
            Hash string
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("Embedding cache hit", text_length=len(text))
            return self._cache[key]

        self._misses += 1
        return None

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        key = self._hash_text(text)

        # Remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Evicted oldest embedding from cache")

        self._cache[key] = embedding
        logger.debug("Cached embedding", text_length=len(text))

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared embedding cache")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }
