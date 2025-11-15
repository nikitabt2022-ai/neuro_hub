"""Dependency injection for API."""

from typing import Optional
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from storage.cache_store import CacheStore
from embeddings.text_embedder import TextEmbedder
from embeddings.embedding_cache import EmbeddingCache
from services.episodic_memory_service import EpisodicMemoryService
from services.semantic_memory_service import SemanticMemoryService
from services.procedural_memory_service import ProceduralMemoryService
from services.working_memory_service import WorkingMemoryService
from services.graph_memory_service import GraphMemoryService


class ServiceContainer:
    """Container for all services."""

    def __init__(self) -> None:
        """Initialize service container."""
        # Storage
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.cache_store: Optional[CacheStore] = None

        # Embeddings
        self.embedder: Optional[TextEmbedder] = None
        self.embedding_cache: Optional[EmbeddingCache] = None

        # Services
        self.episodic_service: Optional[EpisodicMemoryService] = None
        self.semantic_service: Optional[SemanticMemoryService] = None
        self.procedural_service: Optional[ProceduralMemoryService] = None
        self.working_service: Optional[WorkingMemoryService] = None
        self.graph_service: Optional[GraphMemoryService] = None

    async def startup(self) -> None:
        """Initialize all services."""
        # Initialize storage
        self.vector_store = VectorStore()
        self.vector_store.connect()

        self.graph_store = GraphStore()
        self.graph_store.connect()

        self.cache_store = CacheStore()
        await self.cache_store.connect()

        # Initialize embeddings
        self.embedder = TextEmbedder()
        self.embedder.load_model()

        self.embedding_cache = EmbeddingCache()

        # Initialize services
        self.episodic_service = EpisodicMemoryService(
            self.vector_store, self.embedder, self.embedding_cache
        )
        self.semantic_service = SemanticMemoryService(
            self.vector_store, self.embedder, self.embedding_cache
        )
        self.procedural_service = ProceduralMemoryService(
            self.vector_store, self.embedder, self.embedding_cache
        )
        self.working_service = WorkingMemoryService(self.cache_store)
        self.graph_service = GraphMemoryService(self.graph_store)

    async def shutdown(self) -> None:
        """Shutdown all services."""
        if self.vector_store:
            self.vector_store.disconnect()
        if self.graph_store:
            self.graph_store.disconnect()
        if self.cache_store:
            await self.cache_store.disconnect()


# Global service container
_services: Optional[ServiceContainer] = None


def get_services() -> ServiceContainer:
    """Get service container."""
    global _services
    if _services is None:
        _services = ServiceContainer()
    return _services
