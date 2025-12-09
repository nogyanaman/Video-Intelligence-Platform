"""
Audio Intelligence Service - The "Ears"

Uses faster-whisper-large-v3 (Int8) for speech-to-text transcription.
Outputs transcript with millisecond-precision timestamps.

CRITICAL: Uses GPU Governor for VRAM safety.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.exceptions import ProcessingError
from src.governor.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    words: list[dict[str, Any]] | None = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    segments: list[TranscriptSegment]
    language: str
    language_probability: float
    duration_ms: int
    version_hash: str


class AudioIntelligence:
    """
    Audio transcription service using Faster-Whisper.
    
    Model: faster-whisper-large-v3 (Int8 quantization for VRAM efficiency)
    Output: Timestamped transcript segments in milliseconds
    """
    
    MODEL_NAME = "large-v3"
    MODEL_VERSION = "faster-whisper-large-v3-int8"
    
    def __init__(self) -> None:
        """Initialize audio intelligence service."""
        self._model = None
        self._version_hash = self._compute_version_hash()
    
    def _compute_version_hash(self) -> str:
        """Compute version hash for caching."""
        version_string = f"{self.MODEL_VERSION}-v1"
        return hashlib.sha256(version_string.encode()).hexdigest()[:16]
    
    @property
    def version_hash(self) -> str:
        """Get model version hash."""
        return self._version_hash
    
    def _load_model(self) -> Any:
        """Load Faster-Whisper model with Int8 quantization."""
        from faster_whisper import WhisperModel
        
        logger.info(f"Loading Faster-Whisper model: {self.MODEL_NAME}")
        
        model = WhisperModel(
            self.MODEL_NAME,
            device="cuda",
            compute_type="int8",  # Int8 for VRAM efficiency
            download_root=str(settings.models_dir),
        )
        
        logger.info("Faster-Whisper model loaded")
        return model
    
    def _unload_model(self, model: Any) -> None:
        """Unload model and free memory."""
        logger.info("Unloading Faster-Whisper model")
        del model
    
    def transcribe(
        self,
        audio_path: Path,
        video_id: str,
    ) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Uses GPU Governor for safe model loading/unloading.
        
        Args:
            audio_path: Path to WAV audio file
            video_id: Video ID for logging
            
        Returns:
            TranscriptionResult with timestamped segments
        """
        if not audio_path.exists():
            raise ProcessingError(
                message=f"Audio file not found: {audio_path}",
                stage="audio_intelligence",
                video_id=video_id,
            )
        
        # Check if file is empty
        if audio_path.stat().st_size == 0:
            logger.warning(f"Empty audio file for video {video_id}")
            return TranscriptionResult(
                segments=[],
                language="unknown",
                language_probability=0.0,
                duration_ms=0,
                version_hash=self._version_hash,
            )
        
        logger.info(f"Starting transcription for video {video_id}")
        
        segments: list[TranscriptSegment] = []
        language = "unknown"
        language_prob = 0.0
        duration_ms = 0
        
        try:
            # Use GPU Governor context for safe model loading
            with gpu_manager.model_context(
                model_name="faster-whisper",
                loader=self._load_model,
                unloader=self._unload_model,
            ) as model:
                # Transcribe with word-level timestamps
                transcribe_segments, info = model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,  # Voice Activity Detection
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400,
                    ),
                )
                
                language = info.language
                language_prob = info.language_probability
                duration_ms = int(info.duration * 1000)
                
                # Process segments
                for segment in transcribe_segments:
                    # Convert to milliseconds (int64 precision)
                    start_ms = int(segment.start * 1000)
                    end_ms = int(segment.end * 1000)
                    
                    # Extract word-level timestamps if available
                    words = None
                    if segment.words:
                        words = [
                            {
                                "word": word.word,
                                "start_ms": int(word.start * 1000),
                                "end_ms": int(word.end * 1000),
                                "probability": round(word.probability, 4),
                            }
                            for word in segment.words
                        ]
                    
                    segments.append(TranscriptSegment(
                        start_ms=start_ms,
                        end_ms=end_ms,
                        text=segment.text.strip(),
                        confidence=round(segment.avg_logprob, 4) if segment.avg_logprob else 0.0,
                        words=words,
                    ))
                
                logger.info(
                    f"Transcription complete: {len(segments)} segments, "
                    f"language={language} ({language_prob:.2%})"
                )
                
        except Exception as e:
            logger.error(f"Transcription failed for video {video_id}: {e}")
            raise ProcessingError(
                message=f"Transcription failed: {str(e)}",
                stage="audio_intelligence",
                video_id=video_id,
                original_error=str(e),
            ) from e
        
        return TranscriptionResult(
            segments=segments,
            language=language,
            language_probability=round(language_prob, 4),
            duration_ms=duration_ms,
            version_hash=self._version_hash,
        )
    
    def to_dict(self, result: TranscriptionResult) -> dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            "segments": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                    "confidence": seg.confidence,
                    "words": seg.words,
                }
                for seg in result.segments
            ],
            "language": result.language,
            "language_probability": result.language_probability,
            "duration_ms": result.duration_ms,
            "version_hash": result.version_hash,
        }
    
    def get_full_transcript(self, result: TranscriptionResult) -> str:
        """Get full transcript as single string."""
        return " ".join(seg.text for seg in result.segments)