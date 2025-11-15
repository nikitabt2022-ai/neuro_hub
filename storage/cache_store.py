"""Cache store implementation using Redis."""

import json
from typing import Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class CacheStore:
    """Cache store for working memory using Redis."""

    def __init__(self) -> None:
        """Initialize cache store."""
        self.host = settings.redis_host
        self.port = settings.redis_port
        self.db = settings.redis_db
        self.password = settings.redis_password
        self.default_ttl = settings.redis_ttl
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password if self.password else None,
                decode_responses=True,
                max_connections=settings.redis_max_connections,
            )
            # Test connection
            await self._client.ping()
            logger.info("Connected to Redis", host=self.host, port=self.port)
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    def _get_client(self) -> redis.Redis:
        """Get Redis client."""
        if self._client is None:
            raise RuntimeError("Not connected to Redis")
        return self._client

    async def set(
        self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default"
    ) -> None:
        """Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds (None = use default)
            namespace: Namespace for key
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"
        ttl_seconds = ttl if ttl is not None else self.default_ttl

        # Serialize value
        serialized = json.dumps(value, default=str)

        # Set with TTL
        await client.setex(full_key, ttl_seconds, serialized)
        logger.debug(f"Set cache key: {full_key}", ttl=ttl_seconds)

    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get a value from cache.

        Args:
            key: Cache key
            namespace: Namespace for key

        Returns:
            Cached value or None if not found
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"

        value = await client.get(full_key)
        if value is None:
            return None

        # Deserialize
        return json.loads(value)

    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a key from cache.

        Args:
            key: Cache key
            namespace: Namespace for key

        Returns:
            True if deleted, False if not found
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"

        result = await client.delete(full_key)
        logger.debug(f"Deleted cache key: {full_key}", deleted=bool(result))
        return bool(result)

    async def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if a key exists.

        Args:
            key: Cache key
            namespace: Namespace for key

        Returns:
            True if exists, False otherwise
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"
        return bool(await client.exists(full_key))

    async def get_ttl(self, key: str, namespace: str = "default") -> Optional[int]:
        """Get remaining TTL for a key.

        Args:
            key: Cache key
            namespace: Namespace for key

        Returns:
            Remaining TTL in seconds, or None if key doesn't exist
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"
        ttl = await client.ttl(full_key)

        if ttl < 0:
            return None
        return ttl

    async def extend_ttl(self, key: str, additional_seconds: int, namespace: str = "default") -> bool:
        """Extend TTL for a key.

        Args:
            key: Cache key
            additional_seconds: Seconds to add to TTL
            namespace: Namespace for key

        Returns:
            True if extended, False if key doesn't exist
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"

        current_ttl = await client.ttl(full_key)
        if current_ttl < 0:
            return False

        new_ttl = current_ttl + additional_seconds
        return bool(await client.expire(full_key, new_ttl))

    async def list_keys(self, pattern: str = "*", namespace: str = "default") -> List[str]:
        """List keys matching a pattern.

        Args:
            pattern: Key pattern (supports wildcards)
            namespace: Namespace for keys

        Returns:
            List of matching keys (without namespace prefix)
        """
        client = self._get_client()
        full_pattern = f"{namespace}:{pattern}"

        keys = []
        async for key in client.scan_iter(match=full_pattern):
            # Remove namespace prefix
            if key.startswith(f"{namespace}:"):
                keys.append(key[len(namespace) + 1 :])

        return keys

    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.

        Args:
            namespace: Namespace to clear

        Returns:
            Number of keys deleted
        """
        client = self._get_client()
        pattern = f"{namespace}:*"

        count = 0
        async for key in client.scan_iter(match=pattern):
            await client.delete(key)
            count += 1

        logger.info(f"Cleared namespace: {namespace}", count=count)
        return count

    async def increment(self, key: str, amount: int = 1, namespace: str = "default") -> int:
        """Increment a counter.

        Args:
            key: Cache key
            amount: Amount to increment
            namespace: Namespace for key

        Returns:
            New counter value
        """
        client = self._get_client()
        full_key = f"{namespace}:{key}"
        return int(await client.incrby(full_key, amount))

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        client = self._get_client()
        info = await client.info()

        return {
            "used_memory": info.get("used_memory_human"),
            "connected_clients": info.get("connected_clients"),
            "total_commands_processed": info.get("total_commands_processed"),
            "keyspace_hits": info.get("keyspace_hits"),
            "keyspace_misses": info.get("keyspace_misses"),
            "hit_rate": (
                info.get("keyspace_hits", 0)
                / (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                if info.get("keyspace_hits") or info.get("keyspace_misses")
                else 0
            ),
        }
