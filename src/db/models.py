"""
SQLAlchemy Database Models.
Exact implementation as specified in the schema.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class VideoStatus(str, enum.Enum):
    """Video processing status."""
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStage(str, enum.Enum):
    """Pipeline processing stages."""
    INGEST = "ingest"
    NORMALIZATION = "normalization"
    AUDIO_INTELLIGENCE = "audio_intelligence"
    VISUAL_INTELLIGENCE = "visual_intelligence"
    INDEXING = "indexing"


class JobStatus(str, enum.Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class User(Base):
    """User model."""
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    videos: Mapped[list["Video"]] = relationship(
        "Video",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Video(Base):
    """Video model with metadata and processing status."""
    
    __tablename__ = "videos"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus),
        default=VideoStatus.UPLOADING,
        nullable=False,
        index=True,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    original_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    normalized_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )
    checksum_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    original_filename: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    duration_ms: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="videos")
    jobs: Mapped[list["Job"]] = relationship(
        "Job",
        back_populates="video",
        cascade="all, delete-orphan",
    )
    analysis_results: Mapped[list["AnalysisResult"]] = relationship(
        "AnalysisResult",
        back_populates="video",
        cascade="all, delete-orphan",
    )
    timestamps: Mapped[list["Timestamp"]] = relationship(
        "Timestamp",
        back_populates="video",
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        Index("ix_videos_user_status", "user_id", "status"),
    )
    
    def __repr__(self) -> str:
        return f"<Video(id={self.id}, status={self.status})>"


class Job(Base):
    """Job model for tracking pipeline stages."""
    
    __tablename__ = "jobs"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage: Mapped[JobStage] = mapped_column(
        Enum(JobStage),
        nullable=False,
        index=True,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus),
        default=JobStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_log: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    retry_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="jobs")
    
    __table_args__ = (
        Index("ix_jobs_video_stage", "video_id", "stage"),
        UniqueConstraint("video_id", "stage", name="uq_job_video_stage"),
    )
    
    def __repr__(self) -> str:
        return f"<Job(id={self.id}, stage={self.stage}, status={self.status})>"


class AnalysisResult(Base):
    """Analysis results from each processing module."""
    
    __tablename__ = "analysis_results"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    module_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    version_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="analysis_results")
    
    __table_args__ = (
        Index("ix_analysis_results_video_module", "video_id", "module_name"),
        UniqueConstraint("video_id", "module_name", name="uq_analysis_video_module"),
    )
    
    def __repr__(self) -> str:
        return f"<AnalysisResult(id={self.id}, module={self.module_name})>"


class Timestamp(Base):
    """Temporal markers for video events."""
    
    __tablename__ = "timestamps"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    start_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    end_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    label: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    
    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="timestamps")
    
    __table_args__ = (
        Index("ix_timestamps_video_time", "video_id", "start_ms", "end_ms"),
        Index("ix_timestamps_video_event_complex", "video_id", "event_type"),    
    )
    
    def __repr__(self) -> str:
        return f"<Timestamp(id={self.id}, event={self.event_type}, start={self.start_ms}ms)>"