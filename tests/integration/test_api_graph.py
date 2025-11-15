"""Test knowledge graph API endpoints."""

import pytest
from fastapi import status
from uuid import uuid4
from datetime import datetime


class TestGraphEndpoints:
    """Test knowledge graph API endpoints."""

    @pytest.mark.asyncio
    async def test_create_entity_success(self, client, mock_services, sample_entity):
        """Test successfully creating an entity."""
        mock_services.graph_service.create_entity.return_value = sample_entity

        request_data = {
            "name": "John Doe",
            "type": "person",
            "properties": {
                "role": "developer",
                "experience": "5 years",
            },
        }

        response = client.post("/api/v1/graph/entities", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["name"] == request_data["name"]
        assert data["type"] == request_data["type"]

    @pytest.mark.asyncio
    async def test_create_entity_minimal_data(self, client, mock_services):
        """Test creating entity with minimal required fields."""
        entity_data = {
            "id": str(uuid4()),
            "name": "Simple Entity",
            "type": "concept",
            "properties": {},
            "confidence": 1.0,
        }
        mock_services.graph_service.create_entity.return_value = entity_data

        request_data = {
            "name": "Simple Entity",
            "type": "concept",
        }

        response = client.post("/api/v1/graph/entities", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "Simple Entity"

    @pytest.mark.asyncio
    async def test_get_entity_success(self, client, mock_services, sample_entity):
        """Test successfully retrieving an entity."""
        entity_id = sample_entity["id"]
        mock_services.graph_service.get_entity.return_value = sample_entity

        response = client.get(f"/api/v1/graph/entities/{entity_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == entity_id
        assert data["name"] == sample_entity["name"]

    @pytest.mark.asyncio
    async def test_get_entity_not_found(self, client, mock_services):
        """Test retrieving non-existent entity."""
        entity_id = str(uuid4())
        mock_services.graph_service.get_entity.return_value = None

        response = client.get(f"/api/v1/graph/entities/{entity_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_relationship_success(self, client, mock_services, sample_relationship):
        """Test successfully creating a relationship."""
        mock_services.graph_service.create_relationship.return_value = sample_relationship

        request_data = {
            "source_id": sample_relationship["source_id"],
            "target_id": sample_relationship["target_id"],
            "type": "works_with",
            "properties": {
                "since": "2020-01-01",
            },
            "weight": 0.9,
        }

        response = client.post("/api/v1/graph/relationships", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["type"] == request_data["type"]

    @pytest.mark.asyncio
    async def test_create_relationship_minimal_data(self, client, mock_services):
        """Test creating relationship with minimal fields."""
        relationship_data = {
            "id": str(uuid4()),
            "source_id": str(uuid4()),
            "target_id": str(uuid4()),
            "type": "related_to",
            "weight": 0.5,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_services.graph_service.create_relationship.return_value = relationship_data

        request_data = {
            "source_id": relationship_data["source_id"],
            "target_id": relationship_data["target_id"],
            "type": "related_to",
        }

        response = client.post("/api/v1/graph/relationships", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_get_graph_statistics(self, client, mock_services):
        """Test retrieving knowledge graph statistics."""
        stats_data = {
            "total_entities": 1500,
            "total_relationships": 3200,
            "entity_types": {
                "person": 500,
                "concept": 600,
                "organization": 400,
            },
            "relationship_types": {
                "works_with": 1000,
                "related_to": 1200,
                "belongs_to": 1000,
            },
        }
        mock_services.graph_service.get_statistics.return_value = stats_data

        response = client.get("/api/v1/graph/statistics")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_entities" in data
        assert "total_relationships" in data
        assert data["total_entities"] == 1500

    @pytest.mark.asyncio
    async def test_create_entity_with_rich_properties(self, client, mock_services):
        """Test creating entity with complex properties."""
        entity_data = {
            "id": str(uuid4()),
            "name": "TensorFlow",
            "type": "technology",
            "properties": {
                "category": "machine learning",
                "language": "Python",
                "version": "2.15",
                "maintainer": "Google",
                "license": "Apache 2.0",
                "popularity": "high",
            },
            "confidence": 1.0,
        }
        mock_services.graph_service.create_entity.return_value = entity_data

        request_data = {
            "name": "TensorFlow",
            "type": "technology",
            "properties": entity_data["properties"],
        }

        response = client.post("/api/v1/graph/entities", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "properties" in data

    @pytest.mark.asyncio
    async def test_create_entity_error_handling(self, client, mock_services):
        """Test error handling when creating entity."""
        mock_services.graph_service.create_entity.side_effect = Exception("Graph database error")

        request_data = {
            "name": "Test Entity",
            "type": "test",
        }

        response = client.post("/api/v1/graph/entities", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_create_relationship_error_handling(self, client, mock_services):
        """Test error handling when creating relationship."""
        mock_services.graph_service.create_relationship.side_effect = Exception("Connection failed")

        request_data = {
            "source_id": str(uuid4()),
            "target_id": str(uuid4()),
            "type": "test_relation",
        }

        response = client.post("/api/v1/graph/relationships", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_create_bidirectional_relationship(self, client, mock_services):
        """Test creating a bidirectional relationship."""
        relationship_data = {
            "id": str(uuid4()),
            "source_id": str(uuid4()),
            "target_id": str(uuid4()),
            "type": "collaborates_with",
            "properties": {
                "bidirectional": True,
                "project": "AI Research",
            },
            "weight": 0.95,
        }
        mock_services.graph_service.create_relationship.return_value = relationship_data

        request_data = {
            "source_id": relationship_data["source_id"],
            "target_id": relationship_data["target_id"],
            "type": "collaborates_with",
            "properties": relationship_data["properties"],
            "weight": 0.95,
        }

        response = client.post("/api/v1/graph/relationships", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
