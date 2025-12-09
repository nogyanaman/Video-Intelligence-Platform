"""
Video upload and management endpoints.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    JobResponse,
    ObjectDetectionResponse,
    TranscriptResponse,
    VideoListResponse,
    VideoResponse,
    VideoStatusEnum,
    VideoUploadResponse,
)
from src.core.config import settings
from src.core.exceptions import ValidationError
from src.db.models import AnalysisResult, Job, User, Video, VideoStatus
from src.db.session import get_async_session
from src.workers.tasks import process_video_task

router = APIRouter(prefix="/videos", tags=["videos"])


async def get_or_create_user(
    session: AsyncSession,
    email: str = "default@example.com",
) -> User:
    """Get or create a default user."""
    result = await session.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(email=email)
        session.add(user)
        await session.commit()
        await session.refresh(user)
    
    return user


@router.post("", response_model=VideoUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    file: Annotated[UploadFile, File(description="Video file to upload")],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> VideoUploadResponse:
    """
    Upload a video for processing.
    
    Accepts video files up to 500MB and 10 minutes duration.
    Processing happens asynchronously.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )
    
    extension = Path(file.filename).suffix.lower()
    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    
    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )
    
    # Get or create user
    user = await get_or_create_user(session)
    
    # Create video record
    video = Video(
        user_id=user.id,
        status=VideoStatus.UPLOADING,
        original_filename=file.filename,
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)
    
    # Save file to temp directory
    temp_path = settings.temp_dir / f"{video.id}{extension}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check file size while streaming
        total_size = 0
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > settings.max_video_size_bytes:
                    raise ValidationError(
                        message=f"File exceeds maximum size of {settings.max_video_size_mb}MB",
                        field="file",
                    )
                f.write(chunk)
        
        # Queue processing task
        process_video_task.delay(
            video_id=str(video.id),
            user_id=str(user.id),
            input_path=str(temp_path),
            original_extension=extension,
        )
        
        return VideoUploadResponse(
            id=video.id,
            status=VideoStatusEnum.UPLOADING,
            message="Video uploaded successfully. Processing started.",
        )
        
    except ValidationError as e:
        # Cleanup on error
        temp_path.unlink(missing_ok=True)
        video.status = VideoStatus.FAILED
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        video.status = VideoStatus.FAILED
        await session.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        )


@router.get("", response_model=VideoListResponse)
async def list_videos(
    session: Annotated[AsyncSession, Depends(get_async_session)],
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> VideoListResponse:
    """List all videos with pagination."""
    # Get total count
    count_result = await session.execute(select(func.count(Video.id)))
    total = count_result.scalar_one()
    
    # Get videos
    result = await session.execute(
        select(Video)
        .order_by(Video.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    videos = result.scalars().all()
    
    return VideoListResponse(
        videos=[
            VideoResponse(
                id=v.id,
                user_id=v.user_id,
                status=VideoStatusEnum(v.status.value),
                original_filename=v.original_filename,
                file_size_bytes=v.file_size_bytes,
                duration_ms=v.duration_ms,
                metadata=v.metadata_json,
                created_at=v.created_at,
                updated_at=v.updated_at,
            )
            for v in videos
        ],
        total=total,
    )


@router.get("/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> VideoResponse:
    """Get video details by ID."""
    result = await session.execute(
        select(Video).where(Video.id == video_id)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    return VideoResponse(
        id=video.id,
        user_id=video.user_id,
        status=VideoStatusEnum(video.status.value),
        original_filename=video.original_filename,
        file_size_bytes=video.file_size_bytes,
        duration_ms=video.duration_ms,
        metadata=video.metadata_json,
        created_at=video.created_at,
        updated_at=video.updated_at,
    )


@router.get("/{video_id}/jobs", response_model=list[JobResponse])
async def get_video_jobs(
    video_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> list[JobResponse]:
    """Get all processing jobs for a video."""
    result = await session.execute(
        select(Job)
        .where(Job.video_id == video_id)
        .order_by(Job.created_at)
    )
    jobs = result.scalars().all()
    
    return [
        JobResponse(
            id=j.id,
            video_id=j.video_id,
            stage=j.stage.value,
            status=j.status.value,
            error_log=j.error_log,
            created_at=j.created_at,
            started_at=j.started_at,
            completed_at=j.completed_at,
        )
        for j in jobs
    ]


@router.get("/{video_id}/transcript", response_model=TranscriptResponse)
async def get_video_transcript(
    video_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> TranscriptResponse:
    """Get video transcript."""
    result = await session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.video_id == video_id)
        .where(AnalysisResult.module_name == "transcription")
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found. Video may still be processing.",
        )
    
    data = analysis.data
    return TranscriptResponse(
        segments=data.get("segments", []),
        language=data.get("language", "unknown"),
        duration_ms=data.get("duration_ms", 0),
    )


@router.get("/{video_id}/detections", response_model=ObjectDetectionResponse)
async def get_video_detections(
    video_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> ObjectDetectionResponse:
    """Get object detection results."""
    result = await session.execute(
        select(AnalysisResult)
        .where(AnalysisResult.video_id == video_id)
        .where(AnalysisResult.module_name == "object_detection")
    )
    analysis = result.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Object detection results not found. Video may still be processing.",
        )
    
    data = analysis.data
    return ObjectDetectionResponse(
        frames=data.get("frames", []),
        unique_objects=data.get("unique_objects", {}),
    )


@router.delete("/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(
    video_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> None:
    """Delete a video and all associated data."""
    from src.services.indexing import IndexingService
    from src.storage.minio_client import minio_client
    
    result = await session.execute(
        select(Video).where(Video.id == video_id)
    )
    video = result.scalar_one_or_none()
    
    if not video:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video not found",
        )
    
    # Delete from Milvus
    try:
        indexing = IndexingService()
        indexing.delete_video_embeddings(str(video_id))
    except Exception as e:
        pass  # Continue even if Milvus deletion fails
    
    # Delete from MinIO
    try:
        minio_client.delete_video_files(str(video.user_id), str(video_id))
    except Exception as e:
        pass  # Continue even if MinIO deletion fails
    
    # Delete from database (cascade deletes related records)
    await session.delete(video)
    await session.commit()