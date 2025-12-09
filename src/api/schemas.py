"""
Pydantic Schemas for API Request/Response.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# Enums
class VideoStatusEnum(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStageEnum(str, Enum):
    INGEST = "ingest"
    NORMALIZATION = "normalization"
    AUDIO_INTELLIGENCE = "audio_intelligence"
    VISUAL_INTELLIGENCE = "visual_intelligence"
    INDEXING = "indexing"


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# User Schemas
class UserCreate(BaseModel):
    email: EmailStr


class UserResponse(BaseModel):
    id: UUID
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# Video Schemas
class VideoUploadResponse(BaseModel):
    id: UUID
    status: VideoStatusEnum
    message: str


class VideoMetadata(BaseModel):
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    bitrate: int | None = None
    duration_ms: int | None = None
    is_hdr: bool | None = None


class VideoResponse(BaseModel):
    id: UUID
    user_id: UUID
    status: VideoStatusEnum
    original_filename: str | None = None
    file_size_bytes: int | None = None
    duration_ms: int | None = None
    metadata: VideoMetadata | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class VideoListResponse(BaseModel):
    videos: list[VideoResponse]
    total: int


# Job Schemas
class JobResponse(BaseModel):
    id: UUID
    video_id: UUID
    stage: JobStageEnum
    status: JobStatusEnum
    error_log: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        from_attributes = True


# Query Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    video_id: UUID | None = None


class QuerySource(BaseModel):
    video_id: str
    text: str
    start_ms: int
    end_ms: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[QuerySource]
    confidence: float
    cached: bool = False


# Analysis Schemas
class TranscriptSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str
    confidence: float


class TranscriptResponse(BaseModel):
    segments: list[TranscriptSegment]
    language: str
    duration_ms: int


class DetectionResult(BaseModel):
    frame_number: int
    timestamp_ms: int
    detections: list[dict[str, Any]]  # <--- FIXED: Matches the database


class ObjectDetectionResponse(BaseModel):
    frames: list[DetectionResult]
    unique_objects: dict[str, int]


# Health Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    vram_used_gb: float
    vram_total_gb: float
    vram_state: str


class GPUStatusResponse(BaseModel):
    used_gb: float
    total_gb: float
    free_gb: float
    state: str
    utilization_percent: float
    current_model: str | None = None
    is_locked: bool