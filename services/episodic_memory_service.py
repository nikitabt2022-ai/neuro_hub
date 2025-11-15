"""Episodic memory service for conversation history and interactions."""

from typing import List, Optional
from datetime import datetime
from uuid import UUID

from models.memory import MemoryType, Memory, MemoryCreate
from services.base_memory_service import BaseMemoryService
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class EpisodicMemoryService(BaseMemoryService):
    """Service for episodic memory (conversation history, interactions)."""

    def get_memory_type(self) -> MemoryType:
        """Get memory type."""
        return MemoryType.EPISODIC

    async def store_conversation(
        self,
        content: str,
        session_id: str,
        user_id: Optional[str] = None,
        collection: str = "default",
    ) -> Memory:
        """Store a conversation memory.

        Args:
            content: Conversation content
            session_id: Session identifier
            user_id: User identifier
            collection: Collection name

        Returns:
            Created memory
        """
        memory_create = MemoryCreate(
            type=MemoryType.EPISODIC,
            content=content,
            metadata={
                "session_id": session_id,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "category": "conversation",
            },
            collection=collection,
        )

        return await self.store(memory_create)

    async def get_session_history(
        self,
        session_id: str,
        collection: str = "default",
        limit: int = 50,
    ) -> List[Memory]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier
            collection: Collection name
            limit: Maximum results

        Returns:
            List of memories from the session
        """
        # For now, we'll use semantic search with session_id as query
        # In production, you'd want metadata filtering in Milvus
        query = f"session:{session_id}"
        return await self.search(query, collection, limit=limit)

    async def search_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        collection: str = "default",
        limit: int = 50,
    ) -> List[Memory]:
        """Search memories within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            collection: Collection name
            limit: Maximum results

        Returns:
            List of memories in time range
        """
        # This would require temporal indexing in Milvus
        # For now, return placeholder
        logger.warning("Time range search not fully implemented")
        return []
