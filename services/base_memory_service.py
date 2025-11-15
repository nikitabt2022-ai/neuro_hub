"""Base memory service with common functionality."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from models.memory import Memory, MemoryType, MemoryCreate, MemoryUpdate
from storage.vector_store import VectorStore
from embeddings.text_embedder import TextEmbedder
from embeddings.embedding_cache import EmbeddingCache
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class BaseMemoryService(ABC):
    """Base class for memory services."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: TextEmbedder,
        embedding_cache: EmbeddingCache,
    ) -> None:
        """Initialize base memory service.

        Args:
            vector_store: Vector store instance
            embedder: Text embedder instance
            embedding_cache: Embedding cache instance
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.embedding_cache = embedding_cache

    @abstractmethod
    def get_memory_type(self) -> MemoryType:
        """Get the memory type for this service."""
        pass

    def _get_or_compute_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or compute it.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Try cache first
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self.embedder.embed(text)

        # Cache it
        self.embedding_cache.set(text, embedding)

        return embedding

    async def store(self, memory_create: MemoryCreate) -> Memory:
        """Store a new memory.

        Args:
            memory_create: Memory creation data

        Returns:
            Created memory
        """
        # Generate embedding
        embedding = self._get_or_compute_embedding(memory_create.content)

        # Create memory object
        memory = Memory(
            type=self.get_memory_type(),
            content=memory_create.content,
            embedding=embedding,
            metadata=memory_create.metadata or {},
            collection=memory_create.collection,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Store in vector store
        self.vector_store.insert(memory)

        logger.info(
            f"Stored {self.get_memory_type().value} memory",
            memory_id=str(memory.id),
            collection=memory.collection,
        )

        return memory

    async def get(self, memory_id: UUID, collection: str = "default") -> Optional[Memory]:
        """Get a memory by ID.

        Args:
            memory_id: Memory ID
            collection: Collection name

        Returns:
            Memory or None if not found
        """
        result = self.vector_store.get(memory_id, collection, self.get_memory_type())

        if not result:
            return None

        # Convert to Memory object
        memory = Memory(
            id=UUID(result["id"]),
            type=self.get_memory_type(),
            content=result["content"],
            metadata=result["metadata"],
            collection=collection,
            created_at=datetime.fromtimestamp(result["created_at"] / 1000),
            updated_at=datetime.fromtimestamp(result["updated_at"] / 1000),
        )

        return memory

    async def delete(self, memory_id: UUID, collection: str = "default") -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory ID
            collection: Collection name

        Returns:
            True if deleted, False if not found
        """
        deleted = self.vector_store.delete(memory_id, collection, self.get_memory_type())

        if deleted:
            logger.info(
                f"Deleted {self.get_memory_type().value} memory",
                memory_id=str(memory_id),
                collection=collection,
            )

        return deleted

    async def search(
        self,
        query: str,
        collection: str = "default",
        limit: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Memory]:
        """Search for similar memories.

        Args:
            query: Search query
            collection: Collection name
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of matching memories
        """
        # Generate query embedding
        query_embedding = self._get_or_compute_embedding(query)

        # Search vector store
        results = self.vector_store.search(
            embedding=query_embedding,
            collection=collection,
            memory_type=self.get_memory_type(),
            limit=limit,
            min_similarity=min_similarity,
        )

        # Convert to Memory objects
        memories = []
        for result in results:
            memory = Memory(
                id=UUID(result["id"]),
                type=self.get_memory_type(),
                content=result["content"],
                metadata=result["metadata"],
                collection=collection,
                created_at=datetime.fromtimestamp(result["created_at"] / 1000),
                updated_at=datetime.fromtimestamp(result["updated_at"] / 1000),
            )
            memories.append(memory)

        logger.info(
            f"Searched {self.get_memory_type().value} memories",
            query_length=len(query),
            results=len(memories),
            collection=collection,
        )

        return memories
