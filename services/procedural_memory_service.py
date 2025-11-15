"""Procedural memory service for workflows and skills."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from models.memory import MemoryType, Memory, MemoryCreate
from services.base_memory_service import BaseMemoryService
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class ProceduralMemoryService(BaseMemoryService):
    """Service for procedural memory (workflows, skills, behaviors)."""

    def get_memory_type(self) -> MemoryType:
        """Get memory type."""
        return MemoryType.PROCEDURAL

    async def store_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        workflow_id: str,
        collection: str = "default",
    ) -> Memory:
        """Store a workflow.

        Args:
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps
            workflow_id: Workflow identifier
            collection: Collection name

        Returns:
            Created memory
        """
        content = f"{name}: {description}"

        memory_create = MemoryCreate(
            type=MemoryType.PROCEDURAL,
            content=content,
            metadata={
                "workflow_id": workflow_id,
                "workflow_name": name,
                "steps": steps,
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "category": "workflow",
            },
            collection=collection,
        )

        return await self.store(memory_create)

    async def record_execution(
        self,
        memory_id: UUID,
        success: bool,
        execution_time: float,
        collection: str = "default",
    ) -> bool:
        """Record a workflow execution.

        Args:
            memory_id: Workflow memory ID
            success: Whether execution was successful
            execution_time: Execution time in seconds
            collection: Collection name

        Returns:
            True if recorded, False if not found
        """
        memory = await self.get(memory_id, collection)
        if not memory:
            return False

        # Update execution stats
        memory.metadata["execution_count"] = memory.metadata.get("execution_count", 0) + 1

        if success:
            memory.metadata["success_count"] = memory.metadata.get("success_count", 0) + 1
        else:
            memory.metadata["failure_count"] = memory.metadata.get("failure_count", 0) + 1

        # Calculate success rate
        exec_count = memory.metadata["execution_count"]
        success_count = memory.metadata["success_count"]
        memory.metadata["success_rate"] = success_count / exec_count if exec_count > 0 else 0.0

        # Record execution time
        times = memory.metadata.get("execution_times", [])
        times.append(execution_time)
        memory.metadata["execution_times"] = times[-100:]  # Keep last 100
        memory.metadata["avg_execution_time"] = sum(times) / len(times)

        logger.info(
            "Recorded workflow execution",
            memory_id=str(memory_id),
            success=success,
            success_rate=memory.metadata["success_rate"],
        )

        # Note: Would need vector store update operation
        return True

    async def get_best_workflow(
        self,
        task_description: str,
        min_success_rate: float = 0.5,
        collection: str = "default",
    ) -> Optional[Memory]:
        """Find the best workflow for a task.

        Args:
            task_description: Description of the task
            min_success_rate: Minimum success rate threshold
            collection: Collection name

        Returns:
            Best matching workflow or None
        """
        memories = await self.search(
            query=task_description,
            collection=collection,
            limit=10,
        )

        # Filter by success rate and sort
        valid_workflows = [
            m for m in memories
            if m.metadata.get("success_rate", 0.0) >= min_success_rate
        ]

        if not valid_workflows:
            return None

        # Return highest success rate
        return max(valid_workflows, key=lambda m: m.metadata.get("success_rate", 0.0))
