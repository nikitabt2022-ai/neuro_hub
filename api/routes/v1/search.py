"""Search endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query

from models.memory import MemoryType, MemoryResponse
from api.dependencies import get_services, ServiceContainer

router = APIRouter()


@router.get("/", response_model=List[MemoryResponse])
async def search_memories(
    query: str = Query(..., min_length=1),
    memory_type: MemoryType = Query(MemoryType.SEMANTIC),
    collection: str = Query("default"),
    limit: int = Query(10, ge=1, le=100),
    min_similarity: float = Query(0.0, ge=0.0, le=1.0),
    services: ServiceContainer = Depends(get_services),
) -> List[MemoryResponse]:
    """Search for memories across all types."""
    try:
        # Route to appropriate service
        if memory_type == MemoryType.EPISODIC:
            results = await services.episodic_service.search(
                query, collection, limit, min_similarity
            )
        elif memory_type == MemoryType.SEMANTIC:
            results = await services.semantic_service.search(
                query, collection, limit, min_similarity
            )
        elif memory_type == MemoryType.PROCEDURAL:
            results = await services.procedural_service.search(
                query, collection, limit, min_similarity
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid memory type")

        return [MemoryResponse.model_validate(r) for r in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
