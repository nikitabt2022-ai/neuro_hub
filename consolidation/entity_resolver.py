"""Entity resolution for deduplication and merging."""

from typing import List, Dict, Any, Optional
from uuid import UUID
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class EntityResolver:
    """Resolve and deduplicate entities."""

    def __init__(self) -> None:
        """Initialize entity resolver."""
        self.similarity_threshold = settings.entity_resolution_threshold

    def find_similar_entities(
        self,
        entity_name: str,
        existing_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find similar entities.

        Args:
            entity_name: Name of the entity to find matches for
            existing_entities: List of existing entities

        Returns:
            List of similar entities with similarity scores
        """
        similar = []

        for existing in existing_entities:
            similarity = self._calculate_similarity(entity_name, existing.get("name", ""))

            if similarity >= self.similarity_threshold:
                similar.append({
                    "entity": existing,
                    "similarity": similarity,
                })

        # Sort by similarity
        similar.sort(key=lambda x: x["similarity"], reverse=True)

        return similar

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        # Simple similarity using character overlap
        # In production, use more sophisticated methods like Levenshtein or embeddings

        str1_lower = str1.lower()
        str2_lower = str2.lower()

        # Exact match
        if str1_lower == str2_lower:
            return 1.0

        # Substring match
        if str1_lower in str2_lower or str2_lower in str1_lower:
            return 0.9

        # Character overlap
        set1 = set(str1_lower.split())
        set2 = set(str2_lower.split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def resolve_entity(
        self,
        entity_name: str,
        entity_type: str,
        existing_entities: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve an entity to an existing one or create new.

        Args:
            entity_name: Entity name
            entity_type: Entity type
            existing_entities: List of existing entities

        Returns:
            Resolved entity or None if should create new
        """
        # Filter by type
        same_type_entities = [
            e for e in existing_entities
            if e.get("type") == entity_type
        ]

        # Find similar
        similar = self.find_similar_entities(entity_name, same_type_entities)

        if similar and similar[0]["similarity"] >= self.similarity_threshold:
            logger.debug(
                f"Resolved entity to existing: {entity_name}",
                matched=similar[0]["entity"]["name"],
                similarity=similar[0]["similarity"],
            )
            return similar[0]["entity"]

        # No match found, should create new
        return None

    def merge_entities(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge two entities.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Merged entity
        """
        # Keep entity with higher confidence
        primary = entity1 if entity1.get("confidence", 0) >= entity2.get("confidence", 0) else entity2
        secondary = entity2 if primary == entity1 else entity1

        # Merge properties
        merged_properties = primary.get("properties", {}).copy()
        merged_properties.update(secondary.get("properties", {}))

        # Create merged entity
        merged = primary.copy()
        merged["properties"] = merged_properties

        # Combine source IDs
        source_ids = set(primary.get("source_memory_ids", []))
        source_ids.update(secondary.get("source_memory_ids", []))
        merged["source_memory_ids"] = list(source_ids)

        logger.info(
            "Merged entities",
            primary=primary.get("name"),
            secondary=secondary.get("name"),
        )

        return merged
