"""API route modules."""
from src.api.routes.health import router as health_router
from src.api.routes.videos import router as videos_router
from src.api.routes.queries import router as queries_router

__all__ = ["health_router", "videos_router", "queries_router"]