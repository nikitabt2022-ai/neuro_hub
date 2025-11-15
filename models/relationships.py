"""Relationship data models for knowledge graph."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class RelationType(str, Enum):
    """Relationship type enumeration."""

    RELATED_TO = "related_to"
    PART_OF = "part_of"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    KNOWS = "knows"
    CREATED_BY = "created_by"
    INFLUENCED_BY = "influenced_by"
    SIMILAR_TO = "similar_to"
    CAUSED_BY = "caused_by"
    HAPPENED_AT = "happened_at"
    CUSTOM = "custom"


class Relationship(BaseModel):
    """Relationship model for knowledge graph."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4, description="Unique relationship identifier")
    source_id: UUID = Field(description="Source entity ID")
    target_id: UUID = Field(description="Target entity ID")
    type: RelationType = Field(description="Relationship type")
    label: Optional[str] = Field(default=None, description="Custom relationship label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship weight")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from timestamp")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to timestamp")
    source_memory_ids: list[UUID] = Field(
        default_factory=list, description="Source memory IDs"
    )


class RelationshipCreate(BaseModel):
    """Relationship creation request."""

    source_id: UUID = Field(description="Source entity ID")
    target_id: UUID = Field(description="Target entity ID")
    type: RelationType = Field(description="Relationship type")
    label: Optional[str] = Field(default=None, description="Custom relationship label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship weight")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")


class RelationshipUpdate(BaseModel):
    """Relationship update request."""

    label: Optional[str] = Field(default=None, description="Custom relationship label")
    properties: Optional[Dict[str, Any]] = Field(
        default=None, description="Relationship properties"
    )
    weight: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Weight")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")


class RelationshipResponse(BaseModel):
    """Relationship response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="Relationship ID")
    source_id: UUID = Field(description="Source entity ID")
    target_id: UUID = Field(description="Target entity ID")
    type: RelationType = Field(description="Relationship type")
    label: Optional[str] = Field(default=None, description="Custom label")
    properties: Dict[str, Any] = Field(description="Relationship properties")
    weight: float = Field(description="Relationship weight")
    confidence: float = Field(description="Confidence score")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Update timestamp")
    valid_from: Optional[datetime] = Field(default=None, description="Valid from")
    valid_to: Optional[datetime] = Field(default=None, description="Valid to")
