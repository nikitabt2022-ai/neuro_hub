"""Test health check endpoints."""

import pytest
from fastapi import status


class TestHealthEndpoints:
    """Test health check API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert data["docs"] == "/docs"

    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_detailed_health_check_all_services_available(self, client, mock_services):
        """Test detailed health check when all services are available."""
        response = client.get("/api/v1/health/detailed")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["vector_store"] is True
        assert data["services"]["graph_store"] is True
        assert data["services"]["cache_store"] is True
        assert data["services"]["embedder"] is True

    def test_detailed_health_check_degraded_state(self, client, mock_services):
        """Test detailed health check when some services are unavailable."""
        # Simulate missing service
        mock_services.vector_store = None

        response = client.get("/api/v1/health/detailed")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["vector_store"] is False

    def test_health_check_returns_timestamp(self, client):
        """Test that health check includes valid timestamp."""
        response = client.get("/api/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "timestamp" in data
        # Verify timestamp format (ISO 8601)
        from datetime import datetime
        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    def test_health_check_cors_headers(self, client):
        """Test CORS headers on health endpoint."""
        response = client.options("/api/v1/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })

        # CORS should be enabled
        assert "access-control-allow-origin" in response.headers or response.status_code == 200
