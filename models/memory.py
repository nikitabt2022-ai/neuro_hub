"""Memory data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class MemoryType(str, Enum):
    """Memory type enumeration."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class MemoryMetadata(BaseModel):
    """Memory metadata."""

    model_config = ConfigDict(extra="allow")

    tags: List[str] = Field(default_factory=list, description="Memory tags")
    category: Optional[str] = Field(default=None, description="Memory category")
    source: Optional[str] = Field(default=None, description="Memory source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access time")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    custom: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class Memory(BaseModel):
    """Memory base model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique memory identifier")
    type: MemoryType = Field(description="Memory type")
    content: str = Field(description="Memory content")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: MemoryMetadata = Field(
        default_factory=MemoryMetadata, description="Memory metadata"
    )
    collection: str = Field(default="default", description="Collection name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")


class MemoryCreate(BaseModel):
    """Memory creation request."""

    type: MemoryType = Field(description="Memory type")
    content: str = Field(min_length=1, description="Memory content")
    metadata: Optional[MemoryMetadata] = Field(default=None, description="Memory metadata")
    collection: str = Field(default="default", description="Collection name")


class MemoryUpdate(BaseModel):
    """Memory update request."""

    content: Optional[str] = Field(default=None, min_length=1, description="Memory content")
    metadata: Optional[MemoryMetadata] = Field(default=None, description="Memory metadata")


class MemoryResponse(BaseModel):
    """Memory response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="Memory ID")
    type: MemoryType = Field(description="Memory type")
    content: str = Field(description="Memory content")
    metadata: MemoryMetadata = Field(description="Memory metadata")
    collection: str = Field(description="Collection name")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Update timestamp")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score")


class EpisodicMemory(Memory):
    """Episodic memory model."""

    type: MemoryType = Field(default=MemoryType.EPISODIC, frozen=True)
    session_id: str = Field(description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp")
    context: Dict[str, Any] = Field(default_factory=dict, description="Conversation context")


class SemanticMemory(Memory):
    """Semantic memory model."""

    type: MemoryType = Field(default=MemoryType.SEMANTIC, frozen=True)
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    facts: List[str] = Field(default_factory=list, description="Extracted facts")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence")


class ProceduralMemory(Memory):
    """Procedural memory model."""

    type: MemoryType = Field(default=MemoryType.PROCEDURAL, frozen=True)
    workflow_id: str = Field(description="Workflow identifier")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    execution_count: int = Field(default=0, ge=0, description="Execution count")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")


class WorkingMemory(BaseModel):
    """Working memory model (short-term, in-memory)."""

    model_config = ConfigDict(from_attributes=True)

    key: str = Field(description="Memory key")
    value: Any = Field(description="Memory value")
    ttl: int = Field(default=3600, gt=0, description="Time to live in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation time")
    expires_at: datetime = Field(description="Expiration time")
