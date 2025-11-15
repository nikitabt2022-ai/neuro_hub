"""Entity data models for knowledge graph."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class EntityType(str, Enum):
    """Entity type enumeration."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    OBJECT = "object"
    OTHER = "other"


class Entity(BaseModel):
    """Entity model for knowledge graph."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique entity identifier")
    name: str = Field(description="Entity name")
    type: EntityType = Field(description="Entity type")
    description: Optional[str] = Field(default=None, description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    embedding: Optional[List[float]] = Field(default=None, description="Entity embedding")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from timestamp")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to timestamp")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source_memory_ids: List[UUID] = Field(
        default_factory=list, description="Source memory IDs"
    )


class EntityCreate(BaseModel):
    """Entity creation request."""

    name: str = Field(min_length=1, description="Entity name")
    type: EntityType = Field(description="Entity type")
    description: Optional[str] = Field(default=None, description="Entity description")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class EntityUpdate(BaseModel):
    """Entity update request."""

    name: Optional[str] = Field(default=None, min_length=1, description="Entity name")
    type: Optional[EntityType] = Field(default=None, description="Entity type")
    description: Optional[str] = Field(default=None, description="Entity description")
    properties: Optional[Dict[str, Any]] = Field(default=None, description="Entity properties")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence")


class EntityResponse(BaseModel):
    """Entity response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="Entity ID")
    name: str = Field(description="Entity name")
    type: EntityType = Field(description="Entity type")
    description: Optional[str] = Field(default=None, description="Entity description")
    properties: Dict[str, Any] = Field(description="Entity properties")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Update timestamp")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")
    confidence: float = Field(description="Confidence score")
    relationship_count: int = Field(default=0, description="Number of relationships")
