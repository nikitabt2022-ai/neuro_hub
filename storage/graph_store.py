"""Graph store implementation using Neo4j."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from neo4j import GraphDatabase, Driver, Session
from config.settings import settings
from config.logging import StructuredLogger
from models.entities import Entity, EntityType
from models.relationships import Relationship, RelationType

logger = StructuredLogger(__name__)


class GraphStore:
    """Graph store for knowledge graph using Neo4j."""

    def __init__(self) -> None:
        """Initialize graph store."""
        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Connect to Neo4j."""
        try:
            self._driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info("Connected to Neo4j", uri=self.uri)
            # Create indexes
            self._create_indexes()
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    def _get_session(self) -> Session:
        """Get Neo4j session."""
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")
        return self._driver.session(database=self.database)

    def _create_indexes(self) -> None:
        """Create database indexes."""
        with self._get_session() as session:
            # Index on entity ID
            session.run("CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            # Index on entity name
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            # Index on entity type
            session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            logger.debug("Created Neo4j indexes")

    def create_entity(self, entity: Entity) -> None:
        """Create an entity in the graph."""
        with self._get_session() as session:
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                type: $type,
                description: $description,
                properties: $properties,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at),
                valid_from: datetime($valid_from),
                valid_to: datetime($valid_to),
                confidence: $confidence
            })
            RETURN e
            """
            session.run(
                query,
                id=str(entity.id),
                name=entity.name,
                type=entity.type.value,
                description=entity.description,
                properties=entity.properties,
                created_at=entity.created_at.isoformat() if entity.created_at else None,
                updated_at=entity.updated_at.isoformat() if entity.updated_at else None,
                valid_from=entity.valid_from.isoformat() if entity.valid_from else None,
                valid_to=entity.valid_to.isoformat() if entity.valid_to else None,
                confidence=entity.confidence,
            )
            logger.debug(f"Created entity: {entity.name} ({entity.id})")

    def get_entity(self, entity_id: UUID) -> Optional[Dict[str, Any]]:
        """Get an entity by ID."""
        with self._get_session() as session:
            query = "MATCH (e:Entity {id: $id}) RETURN e"
            result = session.run(query, id=str(entity_id))
            record = result.single()

            if not record:
                return None

            entity = record["e"]
            return dict(entity)

    def update_entity(self, entity_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update an entity."""
        with self._get_session() as session:
            # Build SET clause
            set_clauses = []
            params = {"id": str(entity_id), "updated_at": datetime.utcnow().isoformat()}

            for key, value in updates.items():
                if key not in ["id", "created_at"]:
                    set_clauses.append(f"e.{key} = ${key}")
                    params[key] = value

            set_clauses.append("e.updated_at = datetime($updated_at)")

            query = f"""
            MATCH (e:Entity {{id: $id}})
            SET {', '.join(set_clauses)}
            RETURN e
            """

            result = session.run(query, **params)
            return result.single() is not None

    def delete_entity(self, entity_id: UUID) -> bool:
        """Delete an entity and its relationships."""
        with self._get_session() as session:
            query = """
            MATCH (e:Entity {id: $id})
            DETACH DELETE e
            """
            result = session.run(query, id=str(entity_id))
            return result.consume().counters.nodes_deleted > 0

    def create_relationship(self, relationship: Relationship) -> None:
        """Create a relationship between entities."""
        with self._get_session() as session:
            query = """
            MATCH (source:Entity {id: $source_id})
            MATCH (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATED {
                id: $id,
                type: $type,
                label: $label,
                properties: $properties,
                weight: $weight,
                confidence: $confidence,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at),
                valid_from: datetime($valid_from),
                valid_to: datetime($valid_to)
            }]->(target)
            RETURN r
            """
            session.run(
                query,
                id=str(relationship.id),
                source_id=str(relationship.source_id),
                target_id=str(relationship.target_id),
                type=relationship.type.value,
                label=relationship.label,
                properties=relationship.properties,
                weight=relationship.weight,
                confidence=relationship.confidence,
                created_at=relationship.created_at.isoformat(),
                updated_at=relationship.updated_at.isoformat(),
                valid_from=relationship.valid_from.isoformat() if relationship.valid_from else None,
                valid_to=relationship.valid_to.isoformat() if relationship.valid_to else None,
            )
            logger.debug(f"Created relationship: {relationship.id}")

    def get_entity_relationships(
        self, entity_id: UUID, direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        with self._get_session() as session:
            if direction == "outgoing":
                query = """
                MATCH (e:Entity {id: $id})-[r:RELATED]->(target)
                RETURN r, target
                """
            elif direction == "incoming":
                query = """
                MATCH (source:Entity)-[r:RELATED]->(e:Entity {id: $id})
                RETURN r, source
                """
            else:  # both
                query = """
                MATCH (e:Entity {id: $id})-[r:RELATED]-(other)
                RETURN r, other
                """

            result = session.run(query, id=str(entity_id))
            relationships = []

            for record in result:
                rel_data = dict(record["r"])
                other_data = dict(record.get("target") or record.get("source") or record.get("other"))
                relationships.append({"relationship": rel_data, "entity": other_data})

            return relationships

    def find_path(
        self, source_id: UUID, target_id: UUID, max_depth: int = 3
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two entities."""
        with self._get_session() as session:
            query = """
            MATCH path = shortestPath(
                (source:Entity {id: $source_id})-[*..{max_depth}]-(target:Entity {id: $target_id})
            )
            RETURN path
            """
            result = session.run(
                query,
                source_id=str(source_id),
                target_id=str(target_id),
                max_depth=max_depth,
            )
            record = result.single()

            if not record:
                return None

            path = record["path"]
            path_data = []

            for i, node in enumerate(path.nodes):
                path_data.append({"type": "node", "data": dict(node)})
                if i < len(path.relationships):
                    path_data.append({"type": "relationship", "data": dict(path.relationships[i])})

            return path_data

    def search_entities(
        self, name: Optional[str] = None, entity_type: Optional[EntityType] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search for entities."""
        with self._get_session() as session:
            conditions = []
            params: Dict[str, Any] = {"limit": limit}

            if name:
                conditions.append("e.name CONTAINS $name")
                params["name"] = name

            if entity_type:
                conditions.append("e.type = $type")
                params["type"] = entity_type.value

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            query = f"""
            MATCH (e:Entity)
            {where_clause}
            RETURN e
            LIMIT $limit
            """

            result = session.run(query, **params)
            return [dict(record["e"]) for record in result]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._get_session() as session:
            query = """
            MATCH (e:Entity)
            WITH count(e) as entity_count
            MATCH ()-[r:RELATED]->()
            RETURN entity_count, count(r) as relationship_count
            """
            result = session.run(query)
            record = result.single()

            return {
                "entity_count": record["entity_count"] if record else 0,
                "relationship_count": record["relationship_count"] if record else 0,
            }
