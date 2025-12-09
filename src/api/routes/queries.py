"""
Query and search endpoints.
"""
from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import QueryRequest, QueryResponse, QuerySource
from src.db.session import get_async_session
from src.services.intelligence_router import IntelligenceRouter

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post("", response_model=QueryResponse)
async def query_videos(
    request: QueryRequest,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> QueryResponse:
    """
    Query video content using natural language.
    """
    router_service = IntelligenceRouter()
    
    try:
        video_id = str(request.video_id) if request.video_id else None
        
        result = await router_service.query(
            query=request.query,
            video_id=video_id,
        )
        
        return QueryResponse(
            answer=result.answer,
            sources=[
                QuerySource(
                    video_id=s["video_id"],
                    text=s["text"],
                    start_ms=s["start_ms"],
                    end_ms=s["end_ms"],
                    score=s["score"],
                )
                for s in result.sources
            ],
            confidence=result.confidence,
            cached=result.cached,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.get("/search")
async def search_transcripts(
    q: str,
    session: Annotated[AsyncSession, Depends(get_async_session)],
    video_id: UUID | None = None,
    limit: int = 10,
) -> dict:
    """
    Search video transcripts directly.
    """
    from src.services.indexing import IndexingService
    
    indexing = IndexingService()
    
    try:
        video_id_str = str(video_id) if video_id else None
        results = indexing.search_text(
            query=q,
            video_id=video_id_str,
            limit=limit,
        )
        
        return {
            "query": q,
            "results": results,
            "count": len(results),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )