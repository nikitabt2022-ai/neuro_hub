"""Memory operations for consolidation (ADD, UPDATE, DELETE, MERGE, NOOP)."""

from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class MemoryOperation(str, Enum):
    """Memory operation types."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    NOOP = "noop"


class MemoryOperations:
    """Determine and execute memory operations during consolidation."""

    def __init__(self) -> None:
        """Initialize memory operations."""
        self.dedup_threshold = settings.deduplication_threshold

    def determine_operation(
        self,
        new_fact: Dict[str, Any],
        similar_memories: List[Dict[str, Any]],
    ) -> tuple[MemoryOperation, Optional[Dict[str, Any]]]:
        """Determine what operation to perform on a new fact.

        Args:
            new_fact: New fact to consolidate
            similar_memories: List of similar existing memories

        Returns:
            Tuple of (operation, target_memory)
        """
        if not similar_memories:
            # No similar memories, add new
            return MemoryOperation.ADD, None

        # Get most similar memory
        most_similar = similar_memories[0]
        similarity = most_similar.get("similarity_score", 0.0)

        # Very high similarity = duplicate
        if similarity >= self.dedup_threshold:
            # Check if facts contradict
            if self._are_contradictory(new_fact, most_similar):
                # Contradictory facts: update if new is more confident
                if new_fact.get("confidence", 0) > most_similar.get("metadata", {}).get("confidence", 0):
                    return MemoryOperation.UPDATE, most_similar
                else:
                    return MemoryOperation.NOOP, most_similar
            else:
                # Duplicate or reinforcing fact
                return MemoryOperation.NOOP, most_similar

        # Moderate similarity = might need merge or update
        elif similarity >= 0.7:
            # Check if it's an update
            if self._is_update(new_fact, most_similar):
                return MemoryOperation.UPDATE, most_similar
            # Check if should merge
            elif self._should_merge(new_fact, most_similar):
                return MemoryOperation.MERGE, most_similar
            else:
                return MemoryOperation.ADD, None

        # Low similarity = add as new
        else:
            return MemoryOperation.ADD, None

    def _are_contradictory(
        self,
        fact1: Dict[str, Any],
        fact2: Dict[str, Any],
    ) -> bool:
        """Check if two facts contradict each other.

        Args:
            fact1: First fact
            fact2: Second fact

        Returns:
            True if contradictory
        """
        # Simple heuristic: look for negation patterns
        content1 = fact1.get("content", "").lower()
        content2 = fact2.get("content", "").lower()

        negation_words = ["not", "no", "never", "don't", "doesn't", "didn't"]

        has_negation1 = any(word in content1 for word in negation_words)
        has_negation2 = any(word in content2 for word in negation_words)

        # If one has negation and other doesn't, might be contradictory
        if has_negation1 != has_negation2:
            # Check if they're about the same subject
            words1 = set(content1.split())
            words2 = set(content2.split())
            overlap = len(words1 & words2) / max(len(words1), len(words2))

            return overlap > 0.5

        return False

    def _is_update(
        self,
        new_fact: Dict[str, Any],
        existing_memory: Dict[str, Any],
    ) -> bool:
        """Check if new fact is an update to existing memory.

        Args:
            new_fact: New fact
            existing_memory: Existing memory

        Returns:
            True if it's an update
        """
        # Heuristic: if about same subject but with different values
        new_content = new_fact.get("content", "").lower()
        existing_content = existing_memory.get("content", "").lower()

        # Look for update patterns like "now prefers", "changed to", etc.
        update_indicators = ["now", "changed", "updated", "currently", "today"]

        return any(indicator in new_content for indicator in update_indicators)

    def _should_merge(
        self,
        fact1: Dict[str, Any],
        fact2: Dict[str, Any],
    ) -> bool:
        """Check if two facts should be merged.

        Args:
            fact1: First fact
            fact2: Second fact

        Returns:
            True if should merge
        """
        # Merge if they complement each other (add information)
        # Simple heuristic: similar subject, different details
        content1 = fact1.get("content", "").lower()
        content2 = fact2.get("content", "").lower()

        # If one is subset of other, merge
        if content1 in content2 or content2 in content1:
            return len(content1) != len(content2)

        return False

    def execute_operation(
        self,
        operation: MemoryOperation,
        new_fact: Dict[str, Any],
        target_memory: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a memory operation.

        Args:
            operation: Operation to perform
            new_fact: New fact data
            target_memory: Target memory for update/merge/delete

        Returns:
            Result of the operation
        """
        if operation == MemoryOperation.ADD:
            logger.info("Memory operation: ADD", fact=new_fact.get("content", "")[:50])
            return {"operation": "add", "memory": new_fact}

        elif operation == MemoryOperation.UPDATE:
            logger.info(
                "Memory operation: UPDATE",
                memory_id=target_memory.get("id") if target_memory else None,
            )
            # In production, update the memory in storage
            return {"operation": "update", "memory_id": target_memory.get("id"), "new_data": new_fact}

        elif operation == MemoryOperation.DELETE:
            logger.info(
                "Memory operation: DELETE",
                memory_id=target_memory.get("id") if target_memory else None,
            )
            return {"operation": "delete", "memory_id": target_memory.get("id")}

        elif operation == MemoryOperation.MERGE:
            logger.info(
                "Memory operation: MERGE",
                memory_id=target_memory.get("id") if target_memory else None,
            )
            # Merge contents
            merged_content = self._merge_contents(
                new_fact.get("content", ""),
                target_memory.get("content", "") if target_memory else "",
            )
            return {
                "operation": "merge",
                "memory_id": target_memory.get("id") if target_memory else None,
                "merged_content": merged_content,
            }

        else:  # NOOP
            logger.debug("Memory operation: NOOP")
            return {"operation": "noop"}

    def _merge_contents(self, content1: str, content2: str) -> str:
        """Merge two memory contents.

        Args:
            content1: First content
            content2: Second content

        Returns:
            Merged content
        """
        # Simple merge: combine unique information
        if content1 in content2:
            return content2
        elif content2 in content1:
            return content1
        else:
            return f"{content1}. {content2}"
