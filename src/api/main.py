"""
FastAPI Application - API Service.

Handles uploads and query parsing. Stateless CPU-bound service.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src import __version__
from src.api.routes import health_router, queries_router, videos_router
from src.core.config import settings
from src.core.exceptions import VIPException
from src.db.session import db_manager
from src.storage.minio_client import minio_client
from src.storage.redis_client import redis_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{__version__}")
    
    # Initialize database
    await db_manager.init()
    if settings.app_env == "development":
        await db_manager.create_tables()
    
    # Initialize Redis
    await redis_client.init()
    
    # Ensure MinIO buckets exist
    minio_client.ensure_buckets()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    await db_manager.close()
    await redis_client.close()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Local Video Intelligence Platform - GPU-constrained video analysis",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(VIPException)
async def vip_exception_handler(
    request: Request,
    exc: VIPException,
) -> JSONResponse:
    """Handle VIP custom exceptions."""
    return JSONResponse(
        status_code=400,
        content=exc.to_dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle general exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
            }
        },
    )


# Include routers
app.include_router(health_router)
app.include_router(videos_router)
app.include_router(queries_router)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs" if settings.debug else None,
    }