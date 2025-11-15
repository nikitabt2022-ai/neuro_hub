"""Working memory service for short-term, in-memory storage."""

from typing import Any, List, Optional
from datetime import datetime, timedelta

from models.memory import WorkingMemory
from storage.cache_store import CacheStore
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class WorkingMemoryService:
    """Service for working memory (short-term, Redis-backed)."""

    def __init__(self, cache_store: CacheStore) -> None:
        """Initialize working memory service.

        Args:
            cache_store: Cache store instance
        """
        self.cache_store = cache_store

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        namespace: str = "working_memory",
    ) -> WorkingMemory:
        """Set a working memory value.

        Args:
            key: Memory key
            value: Memory value
            ttl: Time to live in seconds
            namespace: Namespace for the key

        Returns:
            Working memory object
        """
        await self.cache_store.set(key, value, ttl=ttl, namespace=namespace)

        created_at = datetime.utcnow()
        expires_at = created_at + timedelta(seconds=ttl)

        logger.debug(
            "Set working memory",
            key=key,
            namespace=namespace,
            ttl=ttl,
        )

        return WorkingMemory(
            key=key,
            value=value,
            ttl=ttl,
            created_at=created_at,
            expires_at=expires_at,
        )

    async def get(self, key: str, namespace: str = "working_memory") -> Optional[Any]:
        """Get a working memory value.

        Args:
            key: Memory key
            namespace: Namespace for the key

        Returns:
            Memory value or None if not found
        """
        value = await self.cache_store.get(key, namespace=namespace)

        if value is not None:
            logger.debug("Retrieved working memory", key=key, namespace=namespace)

        return value

    async def delete(self, key: str, namespace: str = "working_memory") -> bool:
        """Delete a working memory value.

        Args:
            key: Memory key
            namespace: Namespace for the key

        Returns:
            True if deleted, False if not found
        """
        deleted = await self.cache_store.delete(key, namespace=namespace)

        if deleted:
            logger.debug("Deleted working memory", key=key, namespace=namespace)

        return deleted

    async def extend(
        self,
        key: str,
        additional_seconds: int,
        namespace: str = "working_memory",
    ) -> bool:
        """Extend the TTL of a working memory.

        Args:
            key: Memory key
            additional_seconds: Seconds to add to TTL
            namespace: Namespace for the key

        Returns:
            True if extended, False if not found
        """
        extended = await self.cache_store.extend_ttl(
            key, additional_seconds, namespace=namespace
        )

        if extended:
            logger.debug(
                "Extended working memory TTL",
                key=key,
                additional_seconds=additional_seconds,
            )

        return extended

    async def list_keys(
        self,
        pattern: str = "*",
        namespace: str = "working_memory",
    ) -> List[str]:
        """List keys matching a pattern.

        Args:
            pattern: Key pattern (wildcards supported)
            namespace: Namespace for keys

        Returns:
            List of matching keys
        """
        keys = await self.cache_store.list_keys(pattern=pattern, namespace=namespace)
        return keys

    async def clear_namespace(self, namespace: str = "working_memory") -> int:
        """Clear all memories in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted
        """
        count = await self.cache_store.clear_namespace(namespace)
        logger.info("Cleared working memory namespace", namespace=namespace, count=count)
        return count

    async def get_stats(self) -> dict:
        """Get working memory statistics.

        Returns:
            Dictionary with stats
        """
        return await self.cache_store.get_stats()
