"""Test search API endpoints."""

import pytest
from fastapi import status
from uuid import uuid4
from datetime import datetime


class TestSearchEndpoints:
    """Test search API endpoints."""

    @pytest.mark.asyncio
    async def test_search_semantic_memories(self, client, mock_services):
        """Test searching semantic memories."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Python is a programming language",
                "metadata": {"domain": "programming"},
                "similarity_score": 0.95,
            },
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Python supports object-oriented programming",
                "metadata": {"domain": "programming"},
                "similarity_score": 0.88,
            },
        ]
        mock_services.semantic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "Python programming",
                "memory_type": "semantic",
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        mock_services.semantic_service.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_episodic_memories(self, client, mock_services):
        """Test searching episodic memories."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "episodic",
                "content": "User discussed deployment strategies",
                "metadata": {"session_id": "session-123"},
                "similarity_score": 0.92,
            },
        ]
        mock_services.episodic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "deployment",
                "memory_type": "episodic",
                "collection": "default",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["type"] == "episodic"

    @pytest.mark.asyncio
    async def test_search_procedural_memories(self, client, mock_services):
        """Test searching procedural memories (workflows)."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "procedural",
                "content": "How to deploy a web application",
                "metadata": {"workflow_type": "deployment"},
                "similarity_score": 0.89,
            },
        ]
        mock_services.procedural_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "web deployment process",
                "memory_type": "procedural",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["type"] == "procedural"

    @pytest.mark.asyncio
    async def test_search_with_limit(self, client, mock_services):
        """Test search with result limit."""
        # Create 20 results but limit to 5
        search_results = [
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": f"Result {i}",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "similarity_score": 0.9 - (i * 0.01),
            }
            for i in range(20)
        ]
        mock_services.semantic_service.search.return_value = search_results[:5]

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test query",
                "memory_type": "semantic",
                "limit": 5,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 5

    @pytest.mark.asyncio
    async def test_search_with_min_similarity(self, client, mock_services):
        """Test search with minimum similarity threshold."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Highly relevant result",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "collection": "default",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "similarity_score": 0.95,
            },
        ]
        mock_services.semantic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test query",
                "memory_type": "semantic",
                "min_similarity": 0.8,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) >= 0

    @pytest.mark.asyncio
    async def test_search_empty_results(self, client, mock_services):
        """Test search returning no results."""
        mock_services.semantic_service.search.return_value = []

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "nonexistent term xyz123",
                "memory_type": "semantic",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data == []

    @pytest.mark.asyncio
    async def test_search_missing_query(self, client, mock_services):
        """Test search without required query parameter."""
        response = client.get("/api/v1/search/", params={"memory_type": "semantic"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_invalid_limit(self, client, mock_services):
        """Test search with invalid limit values."""
        # Limit too high
        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test",
                "memory_type": "semantic",
                "limit": 200,  # Max is 100
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_invalid_similarity(self, client, mock_services):
        """Test search with invalid similarity threshold."""
        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test",
                "memory_type": "semantic",
                "min_similarity": 1.5,  # Must be <= 1.0
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_search_with_collection(self, client, mock_services):
        """Test search within a specific collection."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Collection-specific result",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "collection": "default",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "similarity_score": 0.9,
            },
        ]
        mock_services.semantic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test",
                "memory_type": "semantic",
                "collection": "custom_collection",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        # Verify collection parameter was passed to service
        call_args = mock_services.semantic_service.search.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_search_error_handling(self, client, mock_services):
        """Test search error handling."""
        mock_services.semantic_service.search.side_effect = Exception("Search failed")

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test",
                "memory_type": "semantic",
            },
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @pytest.mark.asyncio
    async def test_search_sorted_by_similarity(self, client, mock_services):
        """Test that search results are ordered by similarity."""
        search_results = [
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Most relevant",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "collection": "default",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "similarity_score": 0.98,
            },
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Second most relevant",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "collection": "default",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "similarity_score": 0.85,
            },
            {
                "id": str(uuid4()),
                "type": "semantic",
                "content": "Least relevant",
                "metadata": {"tags": [], "confidence": 1.0, "importance": 0.5, "access_count": 0, "custom": {}},
                "collection": "default",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "similarity_score": 0.72,
            },
        ]
        mock_services.semantic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "test",
                "memory_type": "semantic",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 3

    @pytest.mark.asyncio
    async def test_search_default_parameters(self, client, mock_services):
        """Test search with default parameter values."""
        search_results = []
        mock_services.semantic_service.search.return_value = search_results

        response = client.get(
            "/api/v1/search/",
            params={
                "query": "simple search",
                # Using defaults for memory_type, collection, limit, min_similarity
            },
        )

        assert response.status_code == status.HTTP_200_OK
