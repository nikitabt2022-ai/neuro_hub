"""Test data models."""

import pytest
from datetime import datetime
from uuid import UUID

from models.memory import (
    Memory,
    MemoryType,
    MemoryMetadata,
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
)
from models.entities import Entity, EntityType, EntityCreate
from models.relationships import Relationship, RelationType, RelationshipCreate


class TestMemoryModels:
    """Test memory data models."""

    def test_memory_creation(self):
        """Test creating a memory object."""
        metadata = MemoryMetadata(
            tags=["test", "example"],
            category="test_category",
            confidence=0.95,
        )

        memory = Memory(
            type=MemoryType.SEMANTIC,
            content="This is a test memory",
            metadata=metadata,
            collection="test_collection",
        )

        assert memory.id is not None
        assert isinstance(memory.id, UUID)
        assert memory.type == MemoryType.SEMANTIC
        assert memory.content == "This is a test memory"
        assert memory.metadata.tags == ["test", "example"]
        assert memory.metadata.confidence == 0.95
        assert memory.collection == "test_collection"
        assert isinstance(memory.created_at, datetime)

    def test_memory_create_validation(self):
        """Test MemoryCreate validation."""
        # Valid creation
        memory_create = MemoryCreate(
            type=MemoryType.EPISODIC,
            content="Valid content",
            collection="test",
        )
        assert memory_create.type == MemoryType.EPISODIC

        # Invalid - empty content
        with pytest.raises(ValueError):
            MemoryCreate(
                type=MemoryType.SEMANTIC,
                content="",  # Empty content should fail
                collection="test",
            )

    def test_memory_metadata_defaults(self):
        """Test MemoryMetadata default values."""
        metadata = MemoryMetadata()

        assert metadata.tags == []
        assert metadata.category is None
        assert metadata.confidence == 1.0
        assert metadata.importance == 0.5
        assert metadata.access_count == 0


class TestEntityModels:
    """Test entity data models."""

    def test_entity_creation(self):
        """Test creating an entity."""
        entity = Entity(
            name="Test Entity",
            type=EntityType.PERSON,
            description="A test person",
            properties={"age": 30, "location": "SF"},
        )

        assert entity.id is not None
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.PERSON
        assert entity.description == "A test person"
        assert entity.properties["age"] == 30
        assert entity.confidence == 1.0

    def test_entity_create_validation(self):
        """Test EntityCreate validation."""
        entity_create = EntityCreate(
            name="Valid Name",
            type=EntityType.ORGANIZATION,
            description="Test org",
        )

        assert entity_create.name == "Valid Name"
        assert entity_create.type == EntityType.ORGANIZATION

        # Invalid - empty name
        with pytest.raises(ValueError):
            EntityCreate(
                name="",  # Empty name should fail
                type=EntityType.PERSON,
            )


class TestRelationshipModels:
    """Test relationship data models."""

    def test_relationship_creation(self):
        """Test creating a relationship."""
        from uuid import uuid4

        source_id = uuid4()
        target_id = uuid4()

        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            type=RelationType.WORKS_FOR,
            weight=0.9,
            confidence=0.95,
        )

        assert relationship.id is not None
        assert relationship.source_id == source_id
        assert relationship.target_id == target_id
        assert relationship.type == RelationType.WORKS_FOR
        assert relationship.weight == 0.9
        assert relationship.confidence == 0.95

    def test_relationship_weight_validation(self):
        """Test relationship weight bounds."""
        from uuid import uuid4

        # Valid weight
        rel = RelationshipCreate(
            source_id=uuid4(),
            target_id=uuid4(),
            type=RelationType.KNOWS,
            weight=0.5,
        )
        assert 0.0 <= rel.weight <= 1.0

        # Invalid weight - should fail validation
        with pytest.raises(ValueError):
            RelationshipCreate(
                source_id=uuid4(),
                target_id=uuid4(),
                type=RelationType.KNOWS,
                weight=1.5,  # > 1.0
            )
