"""Graph memory service for knowledge graph operations."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from models.entities import Entity, EntityType, EntityCreate
from models.relationships import Relationship, RelationType, RelationshipCreate
from storage.graph_store import GraphStore
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class GraphMemoryService:
    """Service for graph memory (knowledge graph, temporal relationships)."""

    def __init__(self, graph_store: GraphStore) -> None:
        """Initialize graph memory service.

        Args:
            graph_store: Graph store instance
        """
        self.graph_store = graph_store

    async def create_entity(self, entity_create: EntityCreate) -> Entity:
        """Create an entity in the knowledge graph.

        Args:
            entity_create: Entity creation data

        Returns:
            Created entity
        """
        entity = Entity(
            name=entity_create.name,
            type=entity_create.type,
            description=entity_create.description,
            properties=entity_create.properties,
            valid_from=entity_create.valid_from,
            valid_to=entity_create.valid_to,
            confidence=entity_create.confidence,
        )

        self.graph_store.create_entity(entity)

        logger.info("Created entity", entity_id=str(entity.id), name=entity.name)

        return entity

    async def get_entity(self, entity_id: UUID) -> Optional[Entity]:
        """Get an entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity or None if not found
        """
        result = self.graph_store.get_entity(entity_id)

        if not result:
            return None

        # Convert to Entity object
        entity = Entity(
            id=UUID(result["id"]),
            name=result["name"],
            type=EntityType(result["type"]),
            description=result.get("description"),
            properties=result.get("properties", {}),
            confidence=result.get("confidence", 1.0),
        )

        return entity

    async def create_relationship(
        self, relationship_create: RelationshipCreate
    ) -> Relationship:
        """Create a relationship between entities.

        Args:
            relationship_create: Relationship creation data

        Returns:
            Created relationship
        """
        relationship = Relationship(
            source_id=relationship_create.source_id,
            target_id=relationship_create.target_id,
            type=relationship_create.type,
            label=relationship_create.label,
            properties=relationship_create.properties,
            weight=relationship_create.weight,
            confidence=relationship_create.confidence,
            valid_from=relationship_create.valid_from,
            valid_to=relationship_create.valid_to,
        )

        self.graph_store.create_relationship(relationship)

        logger.info(
            "Created relationship",
            relationship_id=str(relationship.id),
            type=relationship.type.value,
        )

        return relationship

    async def get_entity_relationships(
        self, entity_id: UUID, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get all relationships for an entity.

        Args:
            entity_id: Entity ID
            direction: Relationship direction ('outgoing', 'incoming', 'both')

        Returns:
            List of relationships with related entities
        """
        return self.graph_store.get_entity_relationships(entity_id, direction)

    async def find_path(
        self, source_id: UUID, target_id: UUID, max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """Find path between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path depth

        Returns:
            Path data or None if no path found
        """
        path = self.graph_store.find_path(source_id, target_id, max_depth)

        if path:
            logger.info(
                "Found path between entities",
                source_id=str(source_id),
                target_id=str(target_id),
                path_length=len(path),
            )

        return path

    async def search_entities(
        self,
        name: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        limit: int = 50,
    ) -> List[Entity]:
        """Search for entities.

        Args:
            name: Entity name (partial match)
            entity_type: Entity type filter
            limit: Maximum results

        Returns:
            List of matching entities
        """
        results = self.graph_store.search_entities(name, entity_type, limit)

        entities = []
        for result in results:
            entity = Entity(
                id=UUID(result["id"]),
                name=result["name"],
                type=EntityType(result["type"]),
                description=result.get("description"),
                properties=result.get("properties", {}),
                confidence=result.get("confidence", 1.0),
            )
            entities.append(entity)

        logger.info("Searched entities", results=len(entities))

        return entities

    async def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics.

        Returns:
            Dictionary with graph stats
        """
        return self.graph_store.get_statistics()
