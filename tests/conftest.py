"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
from uuid import uuid4


# Mock fixtures for testing without external dependencies

@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    store = Mock()
    store.connect = Mock()
    store.disconnect = Mock()
    store._connected = True

    # Mock insert
    def mock_insert(memory):
        return None

    store.insert = Mock(side_effect=mock_insert)

    # Mock search
    def mock_search(embedding, collection, memory_type, limit=10, min_similarity=0.0, filters=None):
        return [
            {
                "id": str(uuid4()),
                "content": "Mock memory result",
                "metadata": {"tags": ["test"]},
                "similarity_score": 0.9,
                "created_at": 1000000,
                "updated_at": 1000000,
            }
        ]

    store.search = Mock(side_effect=mock_search)

    # Mock get
    def mock_get(memory_id, collection, memory_type):
        return {
            "id": str(memory_id),
            "content": "Mock memory",
            "metadata": {"tags": ["test"]},
            "created_at": 1000000,
            "updated_at": 1000000,
        }

    store.get = Mock(side_effect=mock_get)

    return store


@pytest.fixture
def mock_embedder():
    """Mock embedder for testing."""
    embedder = Mock()

    def mock_embed(text):
        """Return mock embedding."""
        if isinstance(text, str):
            return [0.1] * 384
        return [[0.1] * 384 for _ in text]

    embedder.embed = Mock(side_effect=mock_embed)
    embedder.embed_query = Mock(side_effect=lambda q: [0.1] * 384)
    embedder.get_dimension = Mock(return_value=384)

    return embedder


@pytest.fixture
def mock_embedding_cache():
    """Mock embedding cache for testing."""
    cache = Mock()
    cache._cache = {}

    def mock_get(text):
        return cache._cache.get(text)

    def mock_set(text, embedding):
        cache._cache[text] = embedding

    cache.get = Mock(side_effect=mock_get)
    cache.set = Mock(side_effect=mock_set)

    return cache


@pytest.fixture
def mock_graph_store():
    """Mock graph store for testing."""
    store = Mock()
    store.connect = Mock()
    store.disconnect = Mock()

    def mock_create_entity(entity):
        return None

    def mock_get_entity(entity_id):
        return {
            "id": str(entity_id),
            "name": "Mock Entity",
            "type": "person",
            "properties": {},
        }

    store.create_entity = Mock(side_effect=mock_create_entity)
    store.get_entity = Mock(side_effect=mock_get_entity)

    return store


@pytest.fixture
async def mock_cache_store():
    """Mock cache store for testing."""
    store = Mock()
    store.connect = AsyncMock()
    store.disconnect = AsyncMock()
    store.set = AsyncMock()
    store.get = AsyncMock(return_value=None)
    store.delete = AsyncMock(return_value=True)

    return store


# Sample data fixtures

@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "content": "User prefers dark mode",
        "metadata": {
            "tags": ["preference", "ui"],
            "category": "settings",
            "confidence": 0.95,
        },
        "collection": "user_prefs",
    }


@pytest.fixture
def sample_entity_data():
    """Sample entity data for testing."""
    return {
        "name": "John Doe",
        "type": "person",
        "description": "Software engineer",
        "properties": {"email": "john@example.com"},
    }


@pytest.fixture
def sample_relationship_data():
    """Sample relationship data for testing."""
    return {
        "source_id": uuid4(),
        "target_id": uuid4(),
        "type": "works_for",
        "weight": 0.9,
    }
