"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from config.logging import setup_logging, StructuredLogger
from api.routes.v1 import episodic, semantic, procedural, graph, search, health
from api.dependencies import get_services

logger = StructuredLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting Neuro Hub API", version=settings.app_version)
    setup_logging()

    # Initialize services
    services = get_services()
    await services.startup()

    yield

    # Shutdown
    logger.info("Shutting down Neuro Hub API")
    await services.shutdown()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise-grade AI Long-Term Memory System with MCP integration",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(episodic.router, prefix="/api/v1/episodic", tags=["episodic"])
app.include_router(semantic.router, prefix="/api/v1/semantic", tags=["semantic"])
app.include_router(procedural.router, prefix="/api/v1/procedural", tags=["procedural"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint."""
    return JSONResponse(
        {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": "AI Long-Term Memory System",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
