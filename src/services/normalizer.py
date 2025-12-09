"""
Video Normalizer using FFmpeg.
Handles normalization, HDR tone mapping, and audio extraction.

Uses subprocess for raw FFmpeg commands to ensure proper filter application.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.exceptions import ProcessingError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video technical metadata."""
    codec: str
    width: int
    height: int
    fps: float
    duration_ms: int
    bitrate: int | None
    is_hdr: bool
    color_space: str | None
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    file_size_bytes: int


@dataclass
class NormalizationResult:
    """Result of video normalization."""
    normalized_path: Path
    audio_path: Path
    metadata: VideoMetadata
    checksum: str


class VideoNormalizer:
    """
    Video normalization service using FFmpeg.
    
    Normalization spec:
    - Resolution: 1080p (1920x1080)
    - Frame rate: 30fps (Constant Frame Rate)
    - Codec: H.264
    - HDR Fix: Apply zscale tone mapping for BT.2020 to BT.709
    - Audio: Extract 16-bit PCM WAV at 48kHz
    """
    
    # Target specs
    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080
    TARGET_FPS = 30
    TARGET_AUDIO_RATE = 48000
    
    def __init__(self, temp_dir: Path | None = None) -> None:
        """Initialize normalizer."""
        self.temp_dir = temp_dir or settings.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract video metadata using FFprobe.
        
        Returns comprehensive technical specifications.
        """
        logger.info(f"Extracting metadata from: {video_path}")
        
        if not video_path.exists():
            raise ValidationError(f"Video file not found: {video_path}")
        
        # FFprobe command for JSON output
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            probe_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise ProcessingError(
                message=f"FFprobe failed: {e.stderr}",
                stage="metadata_extraction",
            ) from e
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                message="FFprobe timeout",
                stage="metadata_extraction",
            )
        except json.JSONDecodeError as e:
            raise ProcessingError(
                message=f"Invalid FFprobe output: {e}",
                stage="metadata_extraction",
            )
        
        # Parse video stream
        video_stream = None
        audio_stream = None
        
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video" and video_stream is None:
                video_stream = stream
            elif stream.get("codec_type") == "audio" and audio_stream is None:
                audio_stream = stream
        
        if not video_stream:
            raise ValidationError("No video stream found in file")
        
        # Extract format info
        format_info = probe_data.get("format", {})
        
        # Parse frame rate (can be fraction like "30/1")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)
        
        # Parse duration (in seconds, convert to ms)
        duration_sec = float(format_info.get("duration", video_stream.get("duration", 0)))
        duration_ms = int(duration_sec * 1000)
        
        # Check for HDR (BT.2020 color space)
        color_space = video_stream.get("color_space")
        color_transfer = video_stream.get("color_transfer")
        color_primaries = video_stream.get("color_primaries")
        
        is_hdr = (
            color_space == "bt2020nc" or
            color_primaries == "bt2020" or
            color_transfer in ["smpte2084", "arib-std-b67"]  # PQ or HLG
        )
        
        # Parse audio stream
        audio_codec = None
        audio_sample_rate = None
        audio_channels = None
        
        if audio_stream:
            audio_codec = audio_stream.get("codec_name")
            audio_sample_rate = int(audio_stream.get("sample_rate", 0))
            audio_channels = int(audio_stream.get("channels", 0))
        
        metadata = VideoMetadata(
            codec=video_stream.get("codec_name", "unknown"),
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=round(fps, 2),
            duration_ms=duration_ms,
            bitrate=int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None,
            is_hdr=is_hdr,
            color_space=color_space,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
            file_size_bytes=int(format_info.get("size", 0)),
        )
        
        logger.info(
            f"Metadata extracted: {metadata.width}x{metadata.height} @ {metadata.fps}fps, "
            f"duration: {metadata.duration_ms}ms, HDR: {metadata.is_hdr}"
        )
        
        return metadata
    
    def validate_video(
        self,
        video_path: Path,
        metadata: VideoMetadata | None = None,
    ) -> VideoMetadata:
        """
        Validate video against processing limits.
        
        Limits:
        - Max size: 500MB
        - Max duration: 10 minutes (600 seconds)
        """
        if metadata is None:
            metadata = self.extract_metadata(video_path)
        
        # Check file size
        if metadata.file_size_bytes > settings.max_video_size_bytes:
            raise ValidationError(
                message=f"Video exceeds size limit of {settings.max_video_size_mb}MB",
                field="file_size",
                value=f"{metadata.file_size_bytes / (1024*1024):.1f}MB",
            )
        
        # Check duration
        if metadata.duration_ms > settings.max_video_duration_seconds * 1000:
            raise ValidationError(
                message=f"Video exceeds duration limit of {settings.max_video_duration_seconds}s",
                field="duration",
                value=f"{metadata.duration_ms / 1000:.1f}s",
            )
        
        logger.info("Video validation passed")
        return metadata
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        logger.info(f"Calculating checksum for: {file_path}")
        
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def normalize(
        self,
        input_path: Path,
        output_dir: Path,
        video_id: str,
    ) -> NormalizationResult:
        """
        Normalize video to standard format.
        
        Output:
        - normalized.mp4: 1080p, 30fps CFR, H.264
        - audio.wav: 16-bit PCM, 48kHz
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        normalized_path = output_dir / "normalized.mp4"
        audio_path = output_dir / "audio.wav"
        
        # Extract metadata first
        metadata = self.extract_metadata(input_path)
        
        # Build FFmpeg command with video filters
        video_filters = self._build_video_filters(metadata)
        
        logger.info(f"Normalizing video with filters: {video_filters}")
        
        # FFmpeg command for video normalization
        video_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-vf", video_filters,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-r", str(self.TARGET_FPS),
            "-vsync", "cfr",  # Constant frame rate
            "-pix_fmt", "yuv420p",
            "-an",  # No audio in video file
            str(normalized_path),
        ]
        
        try:
            logger.info("Running video normalization...")
            subprocess.run(
                video_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minute timeout
            )
            logger.info(f"Video normalized: {normalized_path}")
        except subprocess.CalledProcessError as e:
            raise ProcessingError(
                message=f"FFmpeg video normalization failed: {e.stderr}",
                stage="normalization",
                video_id=video_id,
            ) from e
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                message="FFmpeg timeout during normalization",
                stage="normalization",
                video_id=video_id,
            )
        
        # FFmpeg command for audio extraction
        audio_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", str(self.TARGET_AUDIO_RATE),  # 48kHz
            "-ac", "1",  # Mono for AI processing
            str(audio_path),
        ]
        
        try:
            logger.info("Extracting audio...")
            subprocess.run(
                audio_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
            logger.info(f"Audio extracted: {audio_path}")
        except subprocess.CalledProcessError as e:
            # Audio extraction might fail if no audio stream
            logger.warning(f"Audio extraction failed (may be silent video): {e.stderr}")
            # Create empty audio file
            audio_path.touch()
        except subprocess.TimeoutExpired:
            raise ProcessingError(
                message="FFmpeg timeout during audio extraction",
                stage="normalization",
                video_id=video_id,
            )
        
        # Calculate checksum of original
        checksum = self.calculate_checksum(input_path)
        
        return NormalizationResult(
            normalized_path=normalized_path,
            audio_path=audio_path,
            metadata=metadata,
            checksum=checksum,
        )
    
    def _build_video_filters(self, metadata: VideoMetadata) -> str:
        """
        Build FFmpeg video filter chain.
        
        Includes HDR tone mapping if input is HDR.
        """
        filters = []
        
        # HDR to SDR tone mapping using zscale
        if metadata.is_hdr:
            logger.info("Applying HDR to SDR tone mapping")
            # zscale filter for color space conversion
            # BT.2020 -> BT.709 with tone mapping
            filters.extend([
                "zscale=t=linear:npl=100",  # Convert to linear light
                "format=gbrpf32le",
                "zscale=p=bt709",  # Convert primaries to BT.709
                "tonemap=tonemap=hable:desat=0",  # Apply tone mapping
                "zscale=t=bt709:m=bt709:r=tv",  # Convert to BT.709
                "format=yuv420p",
            ])
        
        # Scale to 1080p maintaining aspect ratio
        # pad to exact 1920x1080 if needed
        filters.append(
            f"scale={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:force_original_aspect_ratio=decrease"
        )
        filters.append(
            f"pad={self.TARGET_WIDTH}:{self.TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        )
        
        return ",".join(filters)
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        fps: int = 1,
    ) -> list[Path]:
        """
        Extract frames at specified FPS.
        
        Default: 1 frame per second for object detection.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_pattern = output_dir / "frame_%06d.jpg"
        
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",  # High quality JPEG
            str(frame_pattern),
        ]
        
        try:
            logger.info(f"Extracting frames at {fps} FPS...")
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            raise ProcessingError(
                message=f"Frame extraction failed: {e.stderr}",
                stage="frame_extraction",
            ) from e
        
        # Get list of extracted frames
        frames = sorted(output_dir.glob("frame_*.jpg"))
        logger.info(f"Extracted {len(frames)} frames")
        
        return frames