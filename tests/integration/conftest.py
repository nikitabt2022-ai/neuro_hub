"""Fixtures for API integration tests."""

import sys
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from uuid import UUID, uuid4
from datetime import datetime

# Mock dependencies before importing app
sys.modules['pymilvus'] = MagicMock()
sys.modules['neo4j'] = MagicMock()
redis_mock = MagicMock()
redis_mock.asyncio = MagicMock()
sys.modules['redis'] = redis_mock
sys.modules['redis.asyncio'] = redis_mock.asyncio
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import ServiceContainer, get_services
from models.memory import Memory, MemoryType
from config.settings import settings

# Import routers
from api.routes.v1 import episodic, semantic, procedural, graph, search, health


@asynccontextmanager
async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Test lifespan - no actual startup/shutdown."""
    yield


# Create test app with noop lifespan
test_app = FastAPI(
    title="Test " + settings.app_name,
    version=settings.app_version,
    description="Test API",
    lifespan=test_lifespan,
)

# Add CORS middleware
test_app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers (same as main app)
test_app.include_router(health.router, prefix="/api/v1", tags=["health"])
test_app.include_router(episodic.router, prefix="/api/v1/episodic", tags=["episodic"])
test_app.include_router(semantic.router, prefix="/api/v1/semantic", tags=["semantic"])
test_app.include_router(procedural.router, prefix="/api/v1/procedural", tags=["procedural"])
test_app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
test_app.include_router(search.router, prefix="/api/v1/search", tags=["search"])


@test_app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI Long-Term Memory System",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@pytest.fixture
def mock_services():
    """Create mock service container."""
    services = ServiceContainer()

    # Mock storage
    services.vector_store = Mock()
    services.graph_store = Mock()
    services.cache_store = AsyncMock()

    # Mock embeddings
    services.embedder = Mock()
    services.embedder.embed = Mock(return_value=[0.1] * 384)
    services.embedding_cache = Mock()

    # Mock services with AsyncMock
    services.episodic_service = AsyncMock()
    services.semantic_service = AsyncMock()
    services.procedural_service = AsyncMock()
    services.working_service = AsyncMock()
    services.graph_service = AsyncMock()

    return services


@pytest.fixture
def client(mock_services):
    """Create test client with mocked services."""
    # Override dependency
    test_app.dependency_overrides[get_services] = lambda: mock_services

    # Create client with test app
    with TestClient(test_app) as test_client:
        yield test_client

    # Clean up
    test_app.dependency_overrides.clear()


@pytest.fixture
def sample_memory():
    """Create sample memory for testing."""
    return {
        "id": str(uuid4()),
        "type": "episodic",
        "content": "User discussed project requirements",
        "metadata": {
            "session_id": "test-session-123",
            "tags": ["project", "requirements"],
            "importance": 0.8,
            "confidence": 1.0,
            "access_count": 0,
            "custom": {},
        },
        "collection": "default",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_entity():
    """Create sample entity for testing."""
    return {
        "id": str(uuid4()),
        "name": "John Doe",
        "type": "person",
        "properties": {
            "role": "developer",
            "experience": "5 years",
        },
        "confidence": 1.0,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_relationship():
    """Create sample relationship for testing."""
    return {
        "id": str(uuid4()),
        "source_id": str(uuid4()),
        "target_id": str(uuid4()),
        "type": "works_with",
        "properties": {
            "since": "2020-01-01",
        },
        "weight": 0.9,
        "created_at": datetime.utcnow().isoformat(),
    }
