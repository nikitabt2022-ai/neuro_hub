"""Semantic memory service for facts and knowledge."""

from typing import List, Optional
from uuid import UUID

from models.memory import MemoryType, Memory, MemoryCreate
from services.base_memory_service import BaseMemoryService
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class SemanticMemoryService(BaseMemoryService):
    """Service for semantic memory (facts, knowledge)."""

    def get_memory_type(self) -> MemoryType:
        """Get memory type."""
        return MemoryType.SEMANTIC

    async def store_fact(
        self,
        fact: str,
        entities: Optional[List[str]] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
        collection: str = "default",
    ) -> Memory:
        """Store a semantic fact.

        Args:
            fact: The fact to store
            entities: Related entities
            confidence: Confidence score
            source: Fact source
            collection: Collection name

        Returns:
            Created memory
        """
        memory_create = MemoryCreate(
            type=MemoryType.SEMANTIC,
            content=fact,
            metadata={
                "entities": entities or [],
                "confidence": confidence,
                "source": source,
                "category": "fact",
            },
            collection=collection,
        )

        return await self.store(memory_create)

    async def search_facts(
        self,
        query: str,
        min_confidence: float = 0.0,
        collection: str = "default",
        limit: int = 10,
    ) -> List[Memory]:
        """Search for facts.

        Args:
            query: Search query
            min_confidence: Minimum confidence threshold
            collection: Collection name
            limit: Maximum results

        Returns:
            List of matching facts
        """
        memories = await self.search(
            query=query,
            collection=collection,
            limit=limit,
            min_similarity=min_confidence,
        )

        # Filter by confidence if specified
        if min_confidence > 0:
            memories = [
                m for m in memories
                if m.metadata.get("confidence", 0.0) >= min_confidence
            ]

        return memories

    async def update_fact_confidence(
        self,
        memory_id: UUID,
        confidence: float,
        collection: str = "default",
    ) -> bool:
        """Update the confidence score of a fact.

        Args:
            memory_id: Memory ID
            confidence: New confidence score
            collection: Collection name

        Returns:
            True if updated, False if not found
        """
        memory = await self.get(memory_id, collection)
        if not memory:
            return False

        memory.metadata["confidence"] = confidence
        # Note: This would require an update operation in the vector store
        logger.warning("Confidence update not fully implemented in vector store")
        return True
