"""
MinIO Storage Client.
S3-compatible object storage with bucket structure as specified.

Bucket Structure:
- raw-videos/{user_id}/{video_id}/original.ext
- processed/{user_id}/{video_id}/normalized.mp4
- processed/{user_id}/{video_id}/audio.wav
- frames/{user_id}/{video_id}/*.jpg
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from src.core.config import settings
from src.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class MinIOClient:
    """MinIO storage client with typed methods."""
    
    # Bucket names
    BUCKET_RAW = "raw-videos"
    BUCKET_PROCESSED = "processed"
    BUCKET_FRAMES = "frames"
    
    _instance: MinIOClient | None = None
    
    def __new__(cls) -> MinIOClient:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize MinIO client."""
        if self._initialized:
            return
        
        self._initialized = True
        self._client = Minio(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        
        logger.info(f"MinIO client initialized: {settings.minio_endpoint}")
    
    def ensure_buckets(self) -> None:
        """Create required buckets if they don't exist."""
        for bucket in [self.BUCKET_RAW, self.BUCKET_PROCESSED, self.BUCKET_FRAMES]:
            try:
                if not self._client.bucket_exists(bucket):
                    self._client.make_bucket(bucket)
                    logger.info(f"Created bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Failed to create bucket {bucket}: {e}")
                raise StorageError(f"Failed to create bucket: {bucket}") from e
    
    def _get_object_path(
        self,
        user_id: str,
        video_id: str,
        filename: str,
    ) -> str:
        """Build object path: {user_id}/{video_id}/{filename}"""
        return f"{user_id}/{video_id}/{filename}"
    
    def upload_raw_video(
        self,
        user_id: str,
        video_id: str,
        file_path: Path,
        original_extension: str,
    ) -> str:
        """
        Upload original video to raw-videos bucket.
        Returns the object path.
        """
        object_name = self._get_object_path(
            user_id, video_id, f"original{original_extension}"
        )
        
        try:
            self._client.fput_object(
                bucket_name=self.BUCKET_RAW,
                object_name=object_name,
                file_path=str(file_path),
            )
            logger.info(f"Uploaded raw video: {object_name}")
            return object_name
        except S3Error as e:
            raise StorageError(
                f"Failed to upload raw video",
                bucket=self.BUCKET_RAW,
                object_name=object_name,
            ) from e
    
    def upload_normalized_video(
        self,
        user_id: str,
        video_id: str,
        file_path: Path,
    ) -> str:
        """Upload normalized video (1080p, 30fps, H.264)."""
        object_name = self._get_object_path(user_id, video_id, "normalized.mp4")
        
        try:
            self._client.fput_object(
                bucket_name=self.BUCKET_PROCESSED,
                object_name=object_name,
                file_path=str(file_path),
                content_type="video/mp4",
            )
            logger.info(f"Uploaded normalized video: {object_name}")
            return object_name
        except S3Error as e:
            raise StorageError(
                f"Failed to upload normalized video",
                bucket=self.BUCKET_PROCESSED,
                object_name=object_name,
            ) from e
    
    def upload_audio(
        self,
        user_id: str,
        video_id: str,
        file_path: Path,
    ) -> str:
        """Upload extracted audio (16-bit PCM, 48kHz WAV)."""
        object_name = self._get_object_path(user_id, video_id, "audio.wav")
        
        try:
            self._client.fput_object(
                bucket_name=self.BUCKET_PROCESSED,
                object_name=object_name,
                file_path=str(file_path),
                content_type="audio/wav",
            )
            logger.info(f"Uploaded audio: {object_name}")
            return object_name
        except S3Error as e:
            raise StorageError(
                f"Failed to upload audio",
                bucket=self.BUCKET_PROCESSED,
                object_name=object_name,
            ) from e
    
    def upload_frame(
        self,
        user_id: str,
        video_id: str,
        frame_number: int,
        file_path: Path,
    ) -> str:
        """Upload extracted frame as JPEG."""
        object_name = self._get_object_path(
            user_id, video_id, f"{frame_number:06d}.jpg"
        )
        
        try:
            self._client.fput_object(
                bucket_name=self.BUCKET_FRAMES,
                object_name=object_name,
                file_path=str(file_path),
                content_type="image/jpeg",
            )
            return object_name
        except S3Error as e:
            raise StorageError(
                f"Failed to upload frame",
                bucket=self.BUCKET_FRAMES,
                object_name=object_name,
            ) from e
    
    def upload_frame_bytes(
        self,
        user_id: str,
        video_id: str,
        frame_number: int,
        data: bytes,
    ) -> str:
        """Upload frame from bytes."""
        object_name = self._get_object_path(
            user_id, video_id, f"{frame_number:06d}.jpg"
        )
        
        try:
            self._client.put_object(
                bucket_name=self.BUCKET_FRAMES,
                object_name=object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type="image/jpeg",
            )
            return object_name
        except S3Error as e:
            raise StorageError(
                f"Failed to upload frame bytes",
                bucket=self.BUCKET_FRAMES,
                object_name=object_name,
            ) from e
    
    def download_file(
        self,
        bucket: str,
        object_name: str,
        destination: Path,
    ) -> Path:
        """Download file from MinIO to local path."""
        try:
            self._client.fget_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=str(destination),
            )
            logger.debug(f"Downloaded: {bucket}/{object_name} -> {destination}")
            return destination
        except S3Error as e:
            raise StorageError(
                f"Failed to download file",
                bucket=bucket,
                object_name=object_name,
            ) from e
    
    def get_raw_video_path(
        self,
        user_id: str,
        video_id: str,
        extension: str,
    ) -> str:
        """Get raw video object path."""
        return self._get_object_path(user_id, video_id, f"original{extension}")
    
    def get_normalized_video_path(
        self,
        user_id: str,
        video_id: str,
    ) -> str:
        """Get normalized video object path."""
        return self._get_object_path(user_id, video_id, "normalized.mp4")
    
    def get_audio_path(
        self,
        user_id: str,
        video_id: str,
    ) -> str:
        """Get audio object path."""
        return self._get_object_path(user_id, video_id, "audio.wav")
    
    def delete_video_files(
        self,
        user_id: str,
        video_id: str,
    ) -> None:
        """Delete all files for a video."""
        prefix = f"{user_id}/{video_id}/"
        
        for bucket in [self.BUCKET_RAW, self.BUCKET_PROCESSED, self.BUCKET_FRAMES]:
            try:
                objects = self._client.list_objects(
                    bucket_name=bucket,
                    prefix=prefix,
                    recursive=True,
                )
                for obj in objects:
                    self._client.remove_object(bucket, obj.object_name)
                    logger.debug(f"Deleted: {bucket}/{obj.object_name}")
            except S3Error as e:
                logger.warning(f"Error deleting files from {bucket}: {e}")
    
    def file_exists(self, bucket: str, object_name: str) -> bool:
        """Check if file exists in bucket."""
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False


# Singleton instance
minio_client = MinIOClient()