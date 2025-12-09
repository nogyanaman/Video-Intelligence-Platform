"""
Celery Tasks for Video Processing.

Handles async job execution with proper error handling.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from celery import Celery

from src.core.config import settings
from src.db.session import db_manager
from src.services.pipeline import PipelineOrchestrator

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "video_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_concurrency=1,  # Single worker for GPU
)


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_video_task(
    self,
    video_id: str,
    user_id: str,
    input_path: str,
    original_extension: str,
) -> dict:
    """
    Process video through the full pipeline.
    """
    logger.info(f"Starting video processing task: {video_id}")
    
    async def _process():
        # --- CRITICAL FIX: Reset DB Manager for this new Event Loop ---
        # This prevents "attached to a different loop" errors
        if db_manager._engine is not None:
            await db_manager.close()
        
        # Force a fresh initialization
        db_manager._engine = None
        db_manager._session_factory = None
        await db_manager.init()
        # -------------------------------------------------------------
        
        try:
            async with db_manager.session() as session:
                pipeline = PipelineOrchestrator(session)
                await pipeline.process(
                    video_id=video_id,
                    user_id=user_id,
                    input_path=Path(input_path),
                    original_extension=original_extension,
                )
        finally:
            # Clean up connection before closing the loop
            await db_manager.close()

    try:
        run_async(_process())
        
        return {
            "status": "completed",
            "video_id": video_id,
        }
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task for video {video_id}")
            raise self.retry(exc=e)
        
        return {
            "status": "failed",
            "video_id": video_id,
            "error": str(e),
        }