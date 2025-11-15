"""Test episodic memory API endpoints."""

import pytest
from fastapi import status
from uuid import uuid4


class TestEpisodicMemoryEndpoints:
    """Test episodic memory API endpoints."""

    @pytest.mark.asyncio
    async def test_store_episodic_memory_success(self, client, mock_services, sample_memory):
        """Test successfully storing an episodic memory."""
        # Configure mock to return the memory
        mock_services.episodic_service.store.return_value = sample_memory

        request_data = {
            "type": "episodic",
            "content": "User discussed project requirements",
            "metadata": {
                "session_id": "test-session-123",
                "tags": ["project", "requirements"],
                "importance": 0.8,
            },
        }

        response = client.post("/api/v1/episodic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["content"] == request_data["content"]
        mock_services.episodic_service.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_episodic_memory_minimal_data(self, client, mock_services, sample_memory):
        """Test storing episodic memory with minimal required fields."""
        mock_services.episodic_service.store.return_value = sample_memory

        request_data = {
            "type": "episodic",
            "content": "Minimal episodic memory",
        }

        response = client.post("/api/v1/episodic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["content"] == request_data["content"]

    @pytest.mark.asyncio
    async def test_get_episodic_memory_success(self, client, mock_services, sample_memory):
        """Test successfully retrieving an episodic memory."""
        memory_id = sample_memory["id"]
        mock_services.episodic_service.get.return_value = sample_memory

        response = client.get(f"/api/v1/episodic/{memory_id}?collection=default")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == sample_memory["content"]

    @pytest.mark.asyncio
    async def test_get_episodic_memory_not_found(self, client, mock_services):
        """Test retrieving non-existent episodic memory."""
        memory_id = str(uuid4())
        mock_services.episodic_service.get.return_value = None

        response = client.get(f"/api/v1/episodic/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_episodic_memory_invalid_uuid(self, client, mock_services):
        """Test retrieving memory with invalid UUID format."""
        response = client.get("/api/v1/episodic/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_delete_episodic_memory_success(self, client, mock_services):
        """Test successfully deleting an episodic memory."""
        memory_id = str(uuid4())
        mock_services.episodic_service.delete.return_value = True

        response = client.delete(f"/api/v1/episodic/{memory_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        mock_services.episodic_service.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_episodic_memory_not_found(self, client, mock_services):
        """Test deleting non-existent episodic memory."""
        memory_id = str(uuid4())
        mock_services.episodic_service.delete.return_value = False

        response = client.delete(f"/api/v1/episodic/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_store_episodic_memory_with_error(self, client, mock_services):
        """Test handling service error when storing memory."""
        mock_services.episodic_service.store.side_effect = Exception("Database error")

        request_data = {
            "type": "episodic",
            "content": "Test memory",
        }

        response = client.post("/api/v1/episodic/", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "detail" in response.json()

    @pytest.mark.asyncio
    async def test_store_episodic_memory_with_metadata(self, client, mock_services, sample_memory):
        """Test storing episodic memory with rich metadata."""
        mock_services.episodic_service.store.return_value = sample_memory

        request_data = {
            "type": "episodic",
            "content": "Detailed conversation about AI",
            "metadata": {
                "session_id": "session-456",
                "user_id": "user-123",
                "tags": ["ai", "conversation", "technical"],
                "importance": 0.95,
                "context": "Technical discussion",
                "participants": ["Alice", "Bob"],
            },
        }

        response = client.post("/api/v1/episodic/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_get_episodic_memory_with_custom_collection(self, client, mock_services, sample_memory):
        """Test retrieving memory from custom collection."""
        memory_id = sample_memory["id"]
        collection_name = "custom_collection"
        mock_services.episodic_service.get.return_value = sample_memory

        response = client.get(f"/api/v1/episodic/{memory_id}?collection={collection_name}")

        assert response.status_code == status.HTTP_200_OK
        # Verify collection parameter was passed
        call_args = mock_services.episodic_service.get.call_args
        assert call_args is not None
