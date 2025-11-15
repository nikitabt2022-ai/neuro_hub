"""Search-related data models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict
from models.memory import MemoryType


class HybridSearchParams(BaseModel):
    """Hybrid search parameters."""

    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid search weight (0=BM25, 1=Vector)",
    )
    use_reranking: bool = Field(default=False, description="Enable reranking")
    rerank_top_k: int = Field(default=100, gt=0, description="Candidates for reranking")


class SearchQuery(BaseModel):
    """Search query model."""

    query: str = Field(min_length=1, description="Search query text")
    memory_types: Optional[List[MemoryType]] = Field(
        default=None, description="Filter by memory types"
    )
    collections: Optional[List[str]] = Field(default=None, description="Filter by collections")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")
    category: Optional[str] = Field(default=None, description="Filter by category")
    date_from: Optional[datetime] = Field(default=None, description="Filter from date")
    date_to: Optional[datetime] = Field(default=None, description="Filter to date")
    limit: int = Field(default=10, gt=0, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Results offset")
    min_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    hybrid_params: Optional[HybridSearchParams] = Field(
        default=None, description="Hybrid search parameters"
    )


class SearchResult(BaseModel):
    """Single search result."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(description="Memory ID")
    type: MemoryType = Field(description="Memory type")
    content: str = Field(description="Memory content")
    metadata: Dict[str, Any] = Field(description="Memory metadata")
    collection: str = Field(description="Collection name")
    similarity_score: float = Field(description="Similarity score")
    rank: int = Field(description="Result rank")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Update timestamp")


class SearchResponse(BaseModel):
    """Search response with results."""

    query: str = Field(description="Original query")
    results: List[SearchResult] = Field(description="Search results")
    total_count: int = Field(description="Total matching results")
    search_time_ms: float = Field(description="Search time in milliseconds")
    used_hybrid_search: bool = Field(default=False, description="Used hybrid search")
    used_reranking: bool = Field(default=False, description="Used reranking")


class GraphSearchQuery(BaseModel):
    """Graph search query for knowledge graph."""

    entity_name: Optional[str] = Field(default=None, description="Entity name to search")
    entity_type: Optional[str] = Field(default=None, description="Entity type filter")
    relationship_type: Optional[str] = Field(default=None, description="Relationship type filter")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum traversal depth")
    limit: int = Field(default=50, gt=0, le=500, description="Maximum results")


class GraphPath(BaseModel):
    """Path between entities in knowledge graph."""

    model_config = ConfigDict(from_attributes=True)

    start_entity_id: UUID = Field(description="Start entity ID")
    end_entity_id: UUID = Field(description="End entity ID")
    path_length: int = Field(description="Path length")
    entities: List[Dict[str, Any]] = Field(description="Entities in path")
    relationships: List[Dict[str, Any]] = Field(description="Relationships in path")
    total_weight: float = Field(description="Total path weight")
