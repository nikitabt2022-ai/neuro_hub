"""MCP server implementation for Neuro Hub."""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID
from mcp.server.fastmcp import FastMCP

from config.settings import settings
from config.logging import setup_logging, StructuredLogger
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
from models.memory import MemoryType, MemoryCreate, MemoryMetadata
from models.entities import EntityType, EntityCreate
from models.relationships import RelationType, RelationshipCreate

# Setup logging
setup_logging()
logger = StructuredLogger(__name__)

# Initialize MCP server
mcp = FastMCP(settings.mcp_server_name)

# Global services (initialized on startup)
vector_store: VectorStore = None
graph_store: GraphStore = None
cache_store: CacheStore = None
embedder: TextEmbedder = None
embedding_cache: EmbeddingCache = None
episodic_service: EpisodicMemoryService = None
semantic_service: SemanticMemoryService = None
procedural_service: ProceduralMemoryService = None
working_service: WorkingMemoryService = None
graph_service: GraphMemoryService = None


@mcp.tool()
async def store_memory(
    content: str,
    memory_type: str = "semantic",
    collection: str = "default",
    tags: List[str] = None,
    category: str = None,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Store a new memory in the system.

    Args:
        content: The memory content
        memory_type: Type of memory (episodic, semantic, procedural, working)
        collection: Collection name
        tags: Optional tags
        category: Optional category
        confidence: Confidence score (0-1)

    Returns:
        Created memory details
    """
    try:
        mem_type = MemoryType(memory_type.lower())
    except ValueError:
        return {"error": f"Invalid memory type: {memory_type}"}

    metadata = MemoryMetadata(
        tags=tags or [],
        category=category,
        confidence=confidence,
    )

    memory_create = MemoryCreate(
        type=mem_type,
        content=content,
        metadata=metadata,
        collection=collection,
    )

    # Route to appropriate service
    if mem_type == MemoryType.EPISODIC:
        result = await episodic_service.store(memory_create)
    elif mem_type == MemoryType.SEMANTIC:
        result = await semantic_service.store(memory_create)
    elif mem_type == MemoryType.PROCEDURAL:
        result = await procedural_service.store(memory_create)
    else:
        return {"error": "Working memory not supported via this endpoint"}

    return {
        "id": str(result.id),
        "type": result.type.value,
        "content": result.content,
        "collection": result.collection,
        "created_at": result.created_at.isoformat(),
    }


@mcp.tool()
async def search_memories(
    query: str,
    memory_type: str = "semantic",
    collection: str = "default",
    limit: int = 10,
    min_similarity: float = 0.0,
) -> Dict[str, Any]:
    """Search for memories using semantic similarity.

    Args:
        query: Search query
        memory_type: Type of memory to search
        collection: Collection to search in
        limit: Maximum number of results
        min_similarity: Minimum similarity threshold

    Returns:
        Search results
    """
    try:
        mem_type = MemoryType(memory_type.lower())
    except ValueError:
        return {"error": f"Invalid memory type: {memory_type}"}

    # Route to appropriate service
    if mem_type == MemoryType.EPISODIC:
        results = await episodic_service.search(query, collection, limit, min_similarity)
    elif mem_type == MemoryType.SEMANTIC:
        results = await semantic_service.search(query, collection, limit, min_similarity)
    elif mem_type == MemoryType.PROCEDURAL:
        results = await procedural_service.search(query, collection, limit, min_similarity)
    else:
        return {"error": "Invalid memory type"}

    return {
        "query": query,
        "results": [
            {
                "id": str(r.id),
                "content": r.content,
                "metadata": r.metadata.model_dump(),
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ],
        "count": len(results),
    }


@mcp.tool()
async def get_memory(
    memory_id: str,
    memory_type: str = "semantic",
    collection: str = "default",
) -> Dict[str, Any]:
    """Retrieve a specific memory by ID.

    Args:
        memory_id: Memory ID
        memory_type: Type of memory
        collection: Collection name

    Returns:
        Memory details or error
    """
    try:
        mem_type = MemoryType(memory_type.lower())
        mem_uuid = UUID(memory_id)
    except ValueError as e:
        return {"error": str(e)}

    # Route to appropriate service
    if mem_type == MemoryType.EPISODIC:
        result = await episodic_service.get(mem_uuid, collection)
    elif mem_type == MemoryType.SEMANTIC:
        result = await semantic_service.get(mem_uuid, collection)
    elif mem_type == MemoryType.PROCEDURAL:
        result = await procedural_service.get(mem_uuid, collection)
    else:
        return {"error": "Invalid memory type"}

    if not result:
        return {"error": "Memory not found"}

    return {
        "id": str(result.id),
        "type": result.type.value,
        "content": result.content,
        "metadata": result.metadata.model_dump(),
        "collection": result.collection,
        "created_at": result.created_at.isoformat(),
        "updated_at": result.updated_at.isoformat(),
    }


@mcp.tool()
async def delete_memory(
    memory_id: str,
    memory_type: str = "semantic",
    collection: str = "default",
) -> Dict[str, Any]:
    """Delete a memory.

    Args:
        memory_id: Memory ID
        memory_type: Type of memory
        collection: Collection name

    Returns:
        Success status
    """
    try:
        mem_type = MemoryType(memory_type.lower())
        mem_uuid = UUID(memory_id)
    except ValueError as e:
        return {"error": str(e)}

    # Route to appropriate service
    if mem_type == MemoryType.EPISODIC:
        deleted = await episodic_service.delete(mem_uuid, collection)
    elif mem_type == MemoryType.SEMANTIC:
        deleted = await semantic_service.delete(mem_uuid, collection)
    elif mem_type == MemoryType.PROCEDURAL:
        deleted = await procedural_service.delete(mem_uuid, collection)
    else:
        return {"error": "Invalid memory type"}

    return {"deleted": deleted, "memory_id": memory_id}


@mcp.tool()
async def create_entity(
    name: str,
    entity_type: str,
    description: str = None,
    properties: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Create an entity in the knowledge graph.

    Args:
        name: Entity name
        entity_type: Entity type (person, organization, location, etc.)
        description: Optional description
        properties: Optional additional properties

    Returns:
        Created entity details
    """
    try:
        ent_type = EntityType(entity_type.lower())
    except ValueError:
        return {"error": f"Invalid entity type: {entity_type}"}

    entity_create = EntityCreate(
        name=name,
        type=ent_type,
        description=description,
        properties=properties or {},
    )

    result = await graph_service.create_entity(entity_create)

    return {
        "id": str(result.id),
        "name": result.name,
        "type": result.type.value,
        "description": result.description,
        "created_at": result.created_at.isoformat(),
    }


@mcp.tool()
async def create_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    label: str = None,
    weight: float = 1.0,
) -> Dict[str, Any]:
    """Create a relationship between entities in the knowledge graph.

    Args:
        source_id: Source entity ID
        target_id: Target entity ID
        relationship_type: Type of relationship
        label: Optional custom label
        weight: Relationship weight (0-1)

    Returns:
        Created relationship details
    """
    try:
        rel_type = RelationType(relationship_type.lower())
        source_uuid = UUID(source_id)
        target_uuid = UUID(target_id)
    except ValueError as e:
        return {"error": str(e)}

    rel_create = RelationshipCreate(
        source_id=source_uuid,
        target_id=target_uuid,
        type=rel_type,
        label=label,
        weight=weight,
    )

    result = await graph_service.create_relationship(rel_create)

    return {
        "id": str(result.id),
        "source_id": str(result.source_id),
        "target_id": str(result.target_id),
        "type": result.type.value,
        "weight": result.weight,
        "created_at": result.created_at.isoformat(),
    }


@mcp.tool()
async def search_entities(
    name: str = None,
    entity_type: str = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Search for entities in the knowledge graph.

    Args:
        name: Entity name to search for
        entity_type: Filter by entity type
        limit: Maximum results

    Returns:
        List of matching entities
    """
    ent_type = None
    if entity_type:
        try:
            ent_type = EntityType(entity_type.lower())
        except ValueError:
            return {"error": f"Invalid entity type: {entity_type}"}

    results = await graph_service.search_entities(name, ent_type, limit)

    return {
        "results": [
            {
                "id": str(e.id),
                "name": e.name,
                "type": e.type.value,
                "description": e.description,
            }
            for e in results
        ],
        "count": len(results),
    }


@mcp.tool()
async def get_system_stats() -> Dict[str, Any]:
    """Get system-wide statistics.

    Returns:
        System statistics
    """
    graph_stats = await graph_service.get_statistics()
    cache_stats = await working_service.get_stats()

    return {
        "graph": graph_stats,
        "cache": cache_stats,
        "embedding_cache": embedding_cache.get_stats(),
    }


# Resources
@mcp.resource("memory://{collection}/{memory_type}/list")
async def list_memories(collection: str, memory_type: str) -> str:
    """List all memories in a collection (resource).

    Args:
        collection: Collection name
        memory_type: Memory type

    Returns:
        List of memories as text
    """
    # This would query the vector store for all memories
    # For now, return placeholder
    return f"Listing {memory_type} memories in collection '{collection}'"


@mcp.resource("memory://stats")
async def memory_stats() -> str:
    """Get memory system statistics (resource).

    Returns:
        Statistics as formatted text
    """
    stats = await get_system_stats()
    return f"System Statistics:\n{stats}"


async def initialize_services() -> None:
    """Initialize all services."""
    global vector_store, graph_store, cache_store, embedder, embedding_cache
    global episodic_service, semantic_service, procedural_service, working_service, graph_service

    logger.info("Initializing MCP server services")

    # Initialize storage
    vector_store = VectorStore()
    vector_store.connect()

    graph_store = GraphStore()
    graph_store.connect()

    cache_store = CacheStore()
    await cache_store.connect()

    # Initialize embeddings
    embedder = TextEmbedder()
    embedder.load_model()

    embedding_cache = EmbeddingCache()

    # Initialize services
    episodic_service = EpisodicMemoryService(vector_store, embedder, embedding_cache)
    semantic_service = SemanticMemoryService(vector_store, embedder, embedding_cache)
    procedural_service = ProceduralMemoryService(vector_store, embedder, embedding_cache)
    working_service = WorkingMemoryService(cache_store)
    graph_service = GraphMemoryService(graph_store)

    logger.info("MCP server services initialized successfully")


async def shutdown_services() -> None:
    """Shutdown all services."""
    global vector_store, graph_store, cache_store

    logger.info("Shutting down MCP server services")

    if vector_store:
        vector_store.disconnect()
    if graph_store:
        graph_store.disconnect()
    if cache_store:
        await cache_store.disconnect()


# Lifespan management
@mcp.lifespan()
async def lifespan():
    """MCP server lifespan management."""
    await initialize_services()
    yield
    await shutdown_services()


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
