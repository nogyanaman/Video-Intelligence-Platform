"""
Pipeline Orchestrator.

Executes the 5-stage video processing pipeline in strict sequence:
1. Ingest & Pre-flight
2. Normalization (FFmpeg)
3. Audio Intelligence (Whisper)
4. Visual Intelligence (YOLO)
5. Indexing (Sentence-Transformers + CLIP)

CRITICAL: Enforces aggressive cleanup between stages.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from src.core.config import settings
from src.core.exceptions import ProcessingError, ValidationError
from src.db.models import (
    AnalysisResult,
    Job,
    JobStage,
    JobStatus,
    Timestamp,
    Video,
    VideoStatus,
)
from src.governor.gpu_manager import gpu_manager
from src.services.audio_intelligence import AudioIntelligence
from src.services.indexing import IndexingService
from src.services.normalizer import VideoNormalizer
from src.services.visual_intelligence import VisualIntelligence
from src.storage.minio_client import minio_client
from src.storage.redis_client import redis_client

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the video processing pipeline.
    
    Executes stages strictly sequentially with cleanup between each.
    """
    
    def __init__(self, session: AsyncSession) -> None:
        """Initialize pipeline with database session."""
        self.session = session
        self.normalizer = VideoNormalizer()
        self.audio_intel = AudioIntelligence()
        self.visual_intel = VisualIntelligence()
        self.indexing = IndexingService()
    
    async def _update_video_status(
        self,
        video_id: str,
        status: VideoStatus,
    ) -> None:
        """Update video status in database."""
        await self.session.execute(
            update(Video)
            .where(Video.id == uuid.UUID(video_id))
            .values(status=status, updated_at=datetime.utcnow())
        )
        await self.session.commit()
    
    async def _create_job(
        self,
        video_id: str,
        stage: JobStage,
    ) -> Job:
        """Create a new job for a pipeline stage."""
        job = Job(
            video_id=uuid.UUID(video_id),
            stage=stage,
            status=JobStatus.PENDING,
        )
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job
    
    async def _update_job_status(
        self,
        job: Job,
        status: JobStatus,
        error_log: str | None = None,
    ) -> None:
        """Update job status."""
        job.status = status
        if error_log:
            job.error_log = error_log
        if status == JobStatus.RUNNING:
            job.started_at = datetime.utcnow()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = datetime.utcnow()
        
        await self.session.commit()
    
    async def _save_analysis_result(
        self,
        video_id: str,
        module_name: str,
        data: dict[str, Any],
        version_hash: str,
    ) -> None:
        """Save analysis result to database."""
        result = AnalysisResult(
            video_id=uuid.UUID(video_id),
            module_name=module_name,
            version_hash=version_hash,
            data=data,
        )
        self.session.add(result)
        await self.session.commit()
        
        # Also cache in Redis (ensure redis is initialized first)
        try:
            await redis_client.init()   # <-- ADDED: ensure redis client is initialized
            await redis_client.set_module_result(
                video_id=video_id,
                module_name=module_name,
                version_hash=version_hash,
                result=data,
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    
    async def _save_timestamps(
        self,
        video_id: str,
        timestamps: list[dict[str, Any]],
    ) -> None:
        """Save timestamp markers to database."""
        for ts in timestamps:
            timestamp = Timestamp(
                video_id=uuid.UUID(video_id),
                start_ms=ts["start_ms"],
                end_ms=ts["end_ms"],
                event_type=ts["event_type"],
                confidence=ts["confidence"],
                label=ts["label"],
                metadata_json=ts.get("metadata"),
            )
            self.session.add(timestamp)
        await self.session.commit()
    
    async def process(
        self,
        video_id: str,
        user_id: str,
        input_path: Path,
        original_extension: str,
    ) -> None:
        """
        Execute the full processing pipeline.
        
        Args:
            video_id: Video UUID
            user_id: User UUID
            input_path: Path to uploaded video file
            original_extension: Original file extension
        """
        logger.info(f"Starting pipeline for video {video_id}")
        
        # Create working directory
        work_dir = settings.temp_dir / video_id
        work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            await self._update_video_status(video_id, VideoStatus.PROCESSING)
            
            # Stage 1: Ingest & Pre-flight
            await self._stage_ingest(video_id, user_id, input_path, original_extension, work_dir)
            gpu_manager.force_cleanup()
            
            # Stage 2: Normalization
            normalized_path, audio_path = await self._stage_normalization(
                video_id, user_id, input_path, work_dir
            )
            gpu_manager.force_cleanup()
            
            # Stage 3: Audio Intelligence
            await self._stage_audio_intelligence(video_id, audio_path)
            gpu_manager.force_cleanup()
            
            # Stage 4: Visual Intelligence
            frame_paths = await self._stage_visual_intelligence(
                video_id, normalized_path, work_dir
            )
            gpu_manager.force_cleanup()
            
            # Stage 5: Indexing
            await self._stage_indexing(video_id, frame_paths)
            gpu_manager.force_cleanup()
            
            # Mark as ready
            await self._update_video_status(video_id, VideoStatus.READY)
            logger.info(f"Pipeline complete for video {video_id}")
            
        except Exception as e:
            logger.error(f"Pipeline failed for video {video_id}: {e}")
            await self._update_video_status(video_id, VideoStatus.FAILED)
            raise
            
        finally:
            # Cleanup working directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup work dir: {e}")
            
            # Final GPU cleanup
            gpu_manager.force_cleanup()
    
    async def _stage_ingest(
        self,
        video_id: str,
        user_id: str,
        input_path: Path,
        original_extension: str,
        work_dir: Path,
    ) -> None:
        """
        Stage 1: Ingest & Pre-flight.
        
        - Validate file (Max 500MB, Max 10min)
        - Extract metadata (FFprobe)
        - Calculate SHA-256 checksum
        """
        job = await self._create_job(video_id, JobStage.INGEST)
        await self._update_job_status(job, JobStatus.RUNNING)
        
        try:
            # Validate
            metadata = self.normalizer.validate_video(input_path)
            
            # Calculate checksum
            checksum = self.normalizer.calculate_checksum(input_path)
            
            # Upload to MinIO
            minio_client.ensure_buckets()
            raw_path = minio_client.upload_raw_video(
                user_id=user_id,
                video_id=video_id,
                file_path=input_path,
                original_extension=original_extension,
            )
            
            # Update video record with metadata
            await self.session.execute(
                update(Video)
                .where(Video.id == uuid.UUID(video_id))
                .values(
                    original_path=raw_path,
                    checksum_sha256=checksum,
                    file_size_bytes=metadata.file_size_bytes,
                    duration_ms=metadata.duration_ms,
                    metadata_json={
                        "codec": metadata.codec,
                        "width": metadata.width,
                        "height": metadata.height,
                        "fps": metadata.fps,
                        "bitrate": metadata.bitrate,
                        "is_hdr": metadata.is_hdr,
                        "color_space": metadata.color_space,
                        "audio_codec": metadata.audio_codec,
                        "audio_sample_rate": metadata.audio_sample_rate,
                        "audio_channels": metadata.audio_channels,
                    },
                )
            )
            await self.session.commit()
            
            await self._update_job_status(job, JobStatus.COMPLETED)
            logger.info(f"Stage 1 complete: {video_id}")
            
        except Exception as e:
            await self._update_job_status(job, JobStatus.FAILED, str(e))
            raise ProcessingError(
                message=f"Ingest stage failed: {str(e)}",
                stage="ingest",
                video_id=video_id,
            ) from e
    
    async def _stage_normalization(
        self,
        video_id: str,
        user_id: str,
        input_path: Path,
        work_dir: Path,
    ) -> tuple[Path, Path]:
        """
        Stage 2: Normalization.
        
        - Convert to 1080p, 30fps CFR, H.264
        - HDR to SDR tone mapping if needed
        - Extract audio WAV (16-bit, 48kHz)
        """
        job = await self._create_job(video_id, JobStage.NORMALIZATION)
        await self._update_job_status(job, JobStatus.RUNNING)
        
        try:
            # Normalize video
            result = self.normalizer.normalize(
                input_path=input_path,
                output_dir=work_dir,
                video_id=video_id,
            )
            
            # Upload normalized files to MinIO
            normalized_minio_path = minio_client.upload_normalized_video(
                user_id=user_id,
                video_id=video_id,
                file_path=result.normalized_path,
            )
            
            audio_minio_path = minio_client.upload_audio(
                user_id=user_id,
                video_id=video_id,
                file_path=result.audio_path,
            )
            
            # Update video record
            await self.session.execute(
                update(Video)
                .where(Video.id == uuid.UUID(video_id))
                .values(normalized_path=normalized_minio_path)
            )
            await self.session.commit()
            
            await self._update_job_status(job, JobStatus.COMPLETED)
            logger.info(f"Stage 2 complete: {video_id}")
            
            return result.normalized_path, result.audio_path
            
        except Exception as e:
            await self._update_job_status(job, JobStatus.FAILED, str(e))
            raise ProcessingError(
                message=f"Normalization stage failed: {str(e)}",
                stage="normalization",
                video_id=video_id,
            ) from e
    
    async def _stage_audio_intelligence(
        self,
        video_id: str,
        audio_path: Path,
    ) -> None:
        """
        Stage 3: Audio Intelligence.
        
        - Transcribe using Faster-Whisper
        - Extract timestamps
        """
        job = await self._create_job(video_id, JobStage.AUDIO_INTELLIGENCE)
        await self._update_job_status(job, JobStatus.RUNNING)
        
        try:
            # Transcribe
            result = self.audio_intel.transcribe(audio_path, video_id)
            
            # Save analysis result
            await self._save_analysis_result(
                video_id=video_id,
                module_name="transcription",
                data=self.audio_intel.to_dict(result),
                version_hash=result.version_hash,
            )
            
            # Save timestamps
            timestamps = [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "event_type": "speech",
                    "confidence": seg.confidence,
                    "label": seg.text[:255],  # Truncate for label
                    "metadata": {"full_text": seg.text},
                }
                for seg in result.segments
            ]
            await self._save_timestamps(video_id, timestamps)
            
            await self._update_job_status(job, JobStatus.COMPLETED)
            logger.info(f"Stage 3 complete: {video_id}")
            
        except Exception as e:
            await self._update_job_status(job, JobStatus.FAILED, str(e))
            raise ProcessingError(
                message=f"Audio intelligence stage failed: {str(e)}",
                stage="audio_intelligence",
                video_id=video_id,
            ) from e
    
    async def _stage_visual_intelligence(
        self,
        video_id: str,
        normalized_path: Path,
        work_dir: Path,
    ) -> list[Path]:
        """
        Stage 4: Visual Intelligence.
        
        - Extract frames at 1 FPS
        - Run YOLO object detection
        """
        job = await self._create_job(video_id, JobStage.VISUAL_INTELLIGENCE)
        await self._update_job_status(job, JobStatus.RUNNING)
        
        try:
            # Extract frames
            frames_dir = work_dir / "frames"
            frame_paths = self.normalizer.extract_frames(
                video_path=normalized_path,
                output_dir=frames_dir,
                fps=settings.processing_fps,
            )
            
            # Run object detection
            result = self.visual_intel.analyze_frames(
                frame_paths=frame_paths,
                video_id=video_id,
                fps=settings.processing_fps,
            )
            
            # Save analysis result
            await self._save_analysis_result(
                video_id=video_id,
                module_name="object_detection",
                data=result.to_dict(),
                version_hash=result.version_hash,
            )
            
            # Save object detection timestamps
            timestamps = []
            for frame in result.frames:
                for det in frame.detections:
                    timestamps.append({
                        "start_ms": frame.timestamp_ms,
                        "end_ms": frame.timestamp_ms + 1000,  # 1 second duration
                        "event_type": "object_detection",
                        "confidence": det.confidence,
                        "label": det.class_name,
                        "metadata": {"bbox": det.bbox, "class_id": det.class_id},
                    })
            
            await self._save_timestamps(video_id, timestamps)
            
            await self._update_job_status(job, JobStatus.COMPLETED)
            logger.info(f"Stage 4 complete: {video_id}")
            
            return frame_paths
            
        except Exception as e:
            await self._update_job_status(job, JobStatus.FAILED, str(e))
            raise ProcessingError(
                message=f"Visual intelligence stage failed: {str(e)}",
                stage="visual_intelligence",
                video_id=video_id,
            ) from e
    
    async def _stage_indexing(
        self,
        video_id: str,
        frame_paths: list[Path],
    ) -> None:
        """
        Stage 5: Indexing.
        
        - Vectorize transcript segments
        - Vectorize keyframes
        - Store in Milvus
        """
        job = await self._create_job(video_id, JobStage.INDEXING)
        await self._update_job_status(job, JobStatus.RUNNING)
        
        try:
            # Get transcript segments from database
            result = await self.session.execute(
                select(AnalysisResult)
                .where(AnalysisResult.video_id == uuid.UUID(video_id))
                .where(AnalysisResult.module_name == "transcription")
            )
            analysis = result.scalar_one_or_none()
            
            segments = []
            if analysis and analysis.data:
                segments = analysis.data.get("segments", [])
            
            # Index text
            text_count = self.indexing.index_text(video_id, segments)
            
            # Index images
            image_count = self.indexing.index_images(
                video_id=video_id,
                frame_paths=frame_paths,
                fps=settings.processing_fps,
            )
            
            # Save indexing metadata
            await self._save_analysis_result(
                video_id=video_id,
                module_name="indexing",
                data={
                    "text_embeddings_count": text_count,
                    "image_embeddings_count": image_count,
                },
                version_hash=self.indexing.version_hash,
            )
            
            await self._update_job_status(job, JobStatus.COMPLETED)
            logger.info(f"Stage 5 complete: {video_id}")
            
        except Exception as e:
            await self._update_job_status(job, JobStatus.FAILED, str(e))
            raise ProcessingError(
                message=f"Indexing stage failed: {str(e)}",
                stage="indexing",
                video_id=video_id,
            ) from e