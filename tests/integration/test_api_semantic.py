"""Test semantic memory API endpoints."""

import pytest
from fastapi import status
from uuid import uuid4
from datetime import datetime


class TestSemanticMemoryEndpoints:
    """Test semantic memory (facts) API endpoints."""

    @pytest.mark.asyncio
    async def test_store_semantic_memory_success(self, client, mock_services):
        """Test successfully storing a semantic memory (fact)."""
        fact_data = {
            "id": str(uuid4()),
            "type": "semantic",
            "content": "Python is a high-level programming language",
            "metadata": {
                "category": "programming",
                "confidence": 0.95,
                "tags": [],
                "access_count": 0,
                "importance": 0.5,
                "custom": {},
            },
            "collection": "default",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_services.semantic_service.store.return_value = fact_data

        request_data = {
            "type": "semantic",
            "content": "Python is a high-level programming language",
            "metadata": {
                "category": "programming",
                "confidence": 0.95,
            },
        }

        response = client.post("/api/v1/semantic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["content"] == request_data["content"]

    @pytest.mark.asyncio
    async def test_store_semantic_memory_fact(self, client, mock_services):
        """Test storing factual knowledge."""
        fact_data = {
            "id": str(uuid4()),
            "type": "semantic",
            "content": "The Eiffel Tower is located in Paris",
            "metadata": {
                "domain": "geography",
                "verified": True,
                "tags": [],
                "confidence": 1.0,
                "importance": 0.5,
                "access_count": 0,
                "custom": {},
            },
            "collection": "default",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_services.semantic_service.store.return_value = fact_data

        request_data = {
            "type": "semantic",
            "content": "The Eiffel Tower is located in Paris",
            "metadata": {"domain": "geography", "verified": True},
        }

        response = client.post("/api/v1/semantic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["content"] == request_data["content"]

    @pytest.mark.asyncio
    async def test_get_semantic_memory_success(self, client, mock_services):
        """Test successfully retrieving a semantic memory."""
        memory_id = str(uuid4())
        fact_data = {
            "id": memory_id,
            "type": "semantic",
            "content": "Water boils at 100°C at sea level",
            "metadata": {
                "domain": "physics",
                "tags": [],
                "confidence": 1.0,
                "importance": 0.5,
                "access_count": 0,
                "custom": {},
            },
            "collection": "default",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_services.semantic_service.get.return_value = fact_data

        response = client.get(f"/api/v1/semantic/{memory_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == memory_id
        assert "content" in data

    @pytest.mark.asyncio
    async def test_get_semantic_memory_not_found(self, client, mock_services):
        """Test retrieving non-existent semantic memory."""
        memory_id = str(uuid4())
        mock_services.semantic_service.get.return_value = None

        response = client.get(f"/api/v1/semantic/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_success(self, client, mock_services):
        """Test successfully deleting a semantic memory."""
        memory_id = str(uuid4())
        mock_services.semantic_service.delete.return_value = True

        response = client.delete(f"/api/v1/semantic/{memory_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    @pytest.mark.asyncio
    async def test_delete_semantic_memory_not_found(self, client, mock_services):
        """Test deleting non-existent semantic memory."""
        memory_id = str(uuid4())
        mock_services.semantic_service.delete.return_value = False

        response = client.delete(f"/api/v1/semantic/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_store_semantic_memory_with_relationships(self, client, mock_services):
        """Test storing semantic memory with relationship metadata."""
        fact_data = {
            "id": str(uuid4()),
            "type": "semantic",
            "content": "Albert Einstein developed the theory of relativity",
            "metadata": {
                "domain": "physics",
                "entities": ["Albert Einstein", "theory of relativity"],
                "relationship": "developed",
                "tags": [],
                "confidence": 1.0,
                "importance": 0.5,
                "access_count": 0,
                "custom": {},
            },
            "collection": "default",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        mock_services.semantic_service.store.return_value = fact_data

        request_data = {
            "type": "semantic",
            "content": "Albert Einstein developed the theory of relativity",
            "metadata": {
                "domain": "physics",
                "entities": ["Albert Einstein", "theory of relativity"],
                "relationship": "developed",
            },
        }

        response = client.post("/api/v1/semantic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_store_semantic_memory_error_handling(self, client, mock_services):
        """Test error handling when storing semantic memory."""
        mock_services.semantic_service.store.side_effect = Exception("Storage failed")

        request_data = {
            "type": "semantic",
            "content": "Test fact",
        }

        response = client.post("/api/v1/semantic/", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
