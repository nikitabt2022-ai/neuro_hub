"""Test embedding cache."""

import pytest
from embeddings.embedding_cache import EmbeddingCache


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100)

        assert cache.max_size == 100
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_miss(self):
        """Test cache miss scenario."""
        cache = EmbeddingCache()

        result = cache.get("new text that doesn't exist")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    def test_cache_hit(self):
        """Test cache hit scenario."""
        cache = EmbeddingCache()
        text = "test text"
        embedding = [0.1, 0.2, 0.3] * 128  # 384 dim

        # Set the embedding
        cache.set(text, embedding)

        # Get it back
        result = cache.get(text)

        assert result == embedding
        assert cache._hits == 1
        assert cache._misses == 0

    def test_cache_set_and_get(self):
        """Test setting and getting multiple items."""
        cache = EmbeddingCache()

        # Set multiple embeddings
        cache.set("text1", [0.1] * 384)
        cache.set("text2", [0.2] * 384)
        cache.set("text3", [0.3] * 384)

        # Retrieve them
        assert cache.get("text1") == [0.1] * 384
        assert cache.get("text2") == [0.2] * 384
        assert cache.get("text3") == [0.3] * 384

        assert cache._hits == 3

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache to capacity
        cache.set("text1", [0.1])
        cache.set("text2", [0.2])
        cache.set("text3", [0.3])

        # Add one more (should evict text1)
        cache.set("text4", [0.4])

        # text1 should be evicted
        assert cache.get("text1") is None
        # Others should still be there
        assert cache.get("text2") is not None
        assert cache.get("text3") is not None
        assert cache.get("text4") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache()

        # Some hits
        cache.set("text1", [0.1])
        cache.get("text1")  # hit
        cache.get("text1")  # hit

        # Some misses
        cache.get("text2")  # miss
        cache.get("text3")  # miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.5  # 2 hits / 4 total

    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = EmbeddingCache()

        cache.set("text1", [0.1])
        cache.set("text2", [0.2])

        cache.clear()

        assert cache.get("text1") is None
        assert cache.get("text2") is None
        assert cache._hits == 0
        assert cache._misses == 2
        stats = cache.get_stats()
        assert stats["size"] == 0

    def test_same_text_different_hash(self):
        """Test that same text returns same hash."""
        cache = EmbeddingCache()

        embedding = [0.5] * 384
        cache.set("hello world", embedding)

        # Same text should retrieve same embedding
        result1 = cache.get("hello world")
        result2 = cache.get("hello world")

        assert result1 == embedding
        assert result2 == embedding
        assert cache._hits == 2
