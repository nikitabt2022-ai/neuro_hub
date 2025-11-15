"""Knowledge graph endpoints."""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException

from models.entities import EntityCreate, EntityResponse
from models.relationships import RelationshipCreate, RelationshipResponse
from api.dependencies import get_services, ServiceContainer

router = APIRouter()


@router.post("/entities", response_model=EntityResponse, status_code=201)
async def create_entity(
    entity: EntityCreate,
    services: ServiceContainer = Depends(get_services),
) -> EntityResponse:
    """Create a new entity in the knowledge graph."""
    try:
        result = await services.graph_service.create_entity(entity)
        return EntityResponse.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: UUID,
    services: ServiceContainer = Depends(get_services),
) -> EntityResponse:
    """Get an entity by ID."""
    entity = await services.graph_service.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    return EntityResponse.model_validate(entity)


@router.post("/relationships", response_model=RelationshipResponse, status_code=201)
async def create_relationship(
    relationship: RelationshipCreate,
    services: ServiceContainer = Depends(get_services),
) -> RelationshipResponse:
    """Create a relationship between entities."""
    try:
        result = await services.graph_service.create_relationship(relationship)
        return RelationshipResponse.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_graph_statistics(
    services: ServiceContainer = Depends(get_services),
) -> dict:
    """Get knowledge graph statistics."""
    return await services.graph_service.get_statistics()
