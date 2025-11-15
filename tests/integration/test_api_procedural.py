"""Test procedural memory API endpoints."""

import pytest
from fastapi import status
from uuid import uuid4
from datetime import datetime


class TestProceduralMemoryEndpoints:
    """Test procedural memory (workflows) API endpoints."""

    @pytest.mark.asyncio
    async def test_store_procedural_memory_success(self, client, mock_services):
        """Test successfully storing a procedural memory (workflow)."""
        workflow_data = {
            "id": str(uuid4()),
            "type": "procedural",
            "content": "Step 1: Initialize project. Step 2: Configure settings. Step 3: Deploy",
            "metadata": {
                "workflow_name": "deployment_process",
                "steps": 3,
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
        mock_services.procedural_service.store.return_value = workflow_data

        request_data = {
            "type": "procedural",
            "content": "Step 1: Initialize project. Step 2: Configure settings. Step 3: Deploy",
            "metadata": {
                "workflow_name": "deployment_process",
                "steps": 3,
            },
        }

        response = client.post("/api/v1/procedural/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "id" in data
        assert data["content"] == request_data["content"]

    @pytest.mark.asyncio
    async def test_store_complex_workflow(self, client, mock_services):
        """Test storing a complex multi-step workflow."""
        workflow_data = {
            "id": str(uuid4()),
            "type": "procedural",
            "content": "How to deploy a machine learning model",
            "metadata": {
                "workflow_type": "ml_deployment",
                "steps": ["Train model", "Validate performance", "Containerize", "Deploy to production", "Monitor metrics"],
                "estimated_time": "2 hours",
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
        mock_services.procedural_service.store.return_value = workflow_data

        request_data = {
            "type": "procedural",
            "content": "How to deploy a machine learning model",
            "metadata": {
                "workflow_type": "ml_deployment",
                "steps": ["Train model", "Validate performance", "Containerize", "Deploy to production", "Monitor metrics"],
                "estimated_time": "2 hours",
            },
        }

        response = client.post("/api/v1/procedural/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert "metadata" in data

    @pytest.mark.asyncio
    async def test_get_procedural_memory_success(self, client, mock_services):
        """Test successfully retrieving a procedural memory."""
        memory_id = str(uuid4())
        workflow_data = {
            "id": memory_id,
            "type": "procedural",
            "content": "Standard code review process",
            "metadata": {
                "category": "development",
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
        mock_services.procedural_service.get.return_value = workflow_data

        response = client.get(f"/api/v1/procedural/{memory_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == memory_id

    @pytest.mark.asyncio
    async def test_get_procedural_memory_not_found(self, client, mock_services):
        """Test retrieving non-existent procedural memory."""
        memory_id = str(uuid4())
        mock_services.procedural_service.get.return_value = None

        response = client.get(f"/api/v1/procedural/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_delete_procedural_memory_success(self, client, mock_services):
        """Test successfully deleting a procedural memory."""
        memory_id = str(uuid4())
        mock_services.procedural_service.delete.return_value = True

        response = client.delete(f"/api/v1/procedural/{memory_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    @pytest.mark.asyncio
    async def test_delete_procedural_memory_not_found(self, client, mock_services):
        """Test deleting non-existent procedural memory."""
        memory_id = str(uuid4())
        mock_services.procedural_service.delete.return_value = False

        response = client.delete(f"/api/v1/procedural/{memory_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_store_procedural_memory_with_versioning(self, client, mock_services):
        """Test storing procedural memory with version metadata."""
        workflow_data = {
            "id": str(uuid4()),
            "type": "procedural",
            "content": "CI/CD pipeline configuration",
            "metadata": {
                "version": "2.0",
                "previous_version": "1.5",
                "changes": "Added automated testing step",
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
        mock_services.procedural_service.store.return_value = workflow_data

        request_data = {
            "type": "procedural",
            "content": "CI/CD pipeline configuration",
            "metadata": {
                "version": "2.0",
                "previous_version": "1.5",
                "changes": "Added automated testing step",
            },
        }

        response = client.post("/api/v1/procedural/", json=request_data)

        assert response.status_code == status.HTTP_201_CREATED

    @pytest.mark.asyncio
    async def test_store_procedural_memory_error_handling(self, client, mock_services):
        """Test error handling when storing procedural memory."""
        mock_services.procedural_service.store.side_effect = Exception("Workflow storage failed")

        request_data = {
            "type": "procedural",
            "content": "Test workflow",
        }

        response = client.post("/api/v1/procedural/", json=request_data)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
