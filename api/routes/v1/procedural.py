"""Procedural memory endpoints."""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException

from models.memory import MemoryCreate, MemoryResponse
from api.dependencies import get_services, ServiceContainer

router = APIRouter()


@router.post("/", response_model=MemoryResponse, status_code=201)
async def store_procedural_memory(
    memory: MemoryCreate,
    services: ServiceContainer = Depends(get_services),
) -> MemoryResponse:
    """Store a new procedural memory (workflow)."""
    try:
        result = await services.procedural_service.store(memory)
        return MemoryResponse.model_validate(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_procedural_memory(
    memory_id: UUID,
    collection: str = "default",
    services: ServiceContainer = Depends(get_services),
) -> MemoryResponse:
    """Get a procedural memory by ID."""
    memory = await services.procedural_service.get(memory_id, collection)
    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse.model_validate(memory)


@router.delete("/{memory_id}", status_code=204)
async def delete_procedural_memory(
    memory_id: UUID,
    collection: str = "default",
    services: ServiceContainer = Depends(get_services),
) -> None:
    """Delete a procedural memory."""
    deleted = await services.procedural_service.delete(memory_id, collection)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
