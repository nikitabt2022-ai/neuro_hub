"""Health check endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime

from api.dependencies import get_services, ServiceContainer
from config.settings import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: datetime


class DetailedHealthResponse(BaseModel):
    """Detailed health response."""

    status: str
    version: str
    timestamp: datetime
    services: dict


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(services: ServiceContainer = Depends(get_services)) -> DetailedHealthResponse:
    """Detailed health check with service status."""
    service_status = {
        "vector_store": services.vector_store is not None,
        "graph_store": services.graph_store is not None,
        "cache_store": services.cache_store is not None,
        "embedder": services.embedder is not None,
    }

    return DetailedHealthResponse(
        status="healthy" if all(service_status.values()) else "degraded",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        services=service_status,
    )
