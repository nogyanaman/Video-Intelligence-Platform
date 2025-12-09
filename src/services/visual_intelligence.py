"""
Visual Intelligence Service - The "Eyes"

Uses YOLOv8s (Small) for object detection at 1 FPS.
Outputs detection results with timestamps and bounding boxes.

CRITICAL: Uses GPU Governor for VRAM safety.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.config import settings
from src.core.exceptions import ProcessingError
from src.governor.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


@dataclass
class FrameDetections:
    """Detections for a single frame."""
    frame_number: int
    timestamp_ms: int
    detections: list[Detection] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_number": self.frame_number,
            "timestamp_ms": self.timestamp_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class VisualAnalysisResult:
    """Complete visual analysis result."""
    frames: list[FrameDetections]
    total_frames: int
    unique_objects: dict[str, int]  # class_name -> count
    version_hash: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [f.to_dict() for f in self.frames],
            "total_frames": self.total_frames,
            "unique_objects": self.unique_objects,
            "version_hash": self.version_hash,
        }


class VisualIntelligence:
    """
    Object detection service using YOLOv8.
    
    Model: YOLOv8s (Small variant for VRAM efficiency)
    Processing: 1 FPS (as specified in PROCESSING_FPS)
    """
    
    MODEL_NAME = "yolov8s"
    MODEL_VERSION = "yolov8s-v8.1"
    
    def __init__(self) -> None:
        """Initialize visual intelligence service."""
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
        """Load YOLOv8 model."""
        from ultralytics import YOLO
        
        logger.info(f"Loading YOLO model: {self.MODEL_NAME}")
        
        # YOLOv8s will be downloaded automatically if not present
        model = YOLO(f"{self.MODEL_NAME}.pt")
        
        # Move to GPU
        model.to("cuda")
        
        logger.info("YOLO model loaded")
        return model
    
    def _unload_model(self, model: Any) -> None:
        """Unload model and free memory."""
        logger.info("Unloading YOLO model")
        # Move to CPU before deletion to free GPU memory
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    
    def analyze_frames(
        self,
        frame_paths: list[Path],
        video_id: str,
        fps: int = 1,
    ) -> VisualAnalysisResult:
        """
        Analyze frames for object detection.
        
        Uses GPU Governor for safe model loading/unloading.
        Processes frames SEQUENTIALLY (no parallel processing).
        
        Args:
            frame_paths: List of paths to frame images
            video_id: Video ID for logging
            fps: Original extraction FPS for timestamp calculation
            
        Returns:
            VisualAnalysisResult with detections per frame
        """
        if not frame_paths:
            logger.warning(f"No frames to analyze for video {video_id}")
            return VisualAnalysisResult(
                frames=[],
                total_frames=0,
                unique_objects={},
                version_hash=self._version_hash,
            )
        
        logger.info(f"Starting visual analysis for video {video_id}: {len(frame_paths)} frames")
        
        frame_results: list[FrameDetections] = []
        object_counts: dict[str, int] = {}
        
        try:
            # Use GPU Governor context for safe model loading
            with gpu_manager.model_context(
                model_name="yolov8",
                loader=self._load_model,
                unloader=self._unload_model,
            ) as model:
                
                # Process frames SEQUENTIALLY (Rule 2: Strict Seriality)
                for idx, frame_path in enumerate(frame_paths):
                    if not frame_path.exists():
                        logger.warning(f"Frame not found: {frame_path}")
                        continue
                    
                    # Calculate timestamp in milliseconds
                    # At 1 FPS, frame 0 = 0ms, frame 1 = 1000ms, etc.
                    timestamp_ms = int((idx / fps) * 1000)
                    
                    try:
                        # Run inference
                        results = model(
                            str(frame_path),
                            verbose=False,
                            conf=0.25,  # Confidence threshold
                        )
                        
                        detections: list[Detection] = []
                        
                        for result in results:
                            if result.boxes is None:
                                continue
                            
                            boxes = result.boxes
                            
                            for i in range(len(boxes)):
                                # Get normalized bbox
                                xyxyn = boxes.xyxyn[i].cpu().numpy()
                                
                                class_id = int(boxes.cls[i].cpu().numpy())
                                class_name = model.names[class_id]
                                confidence = float(boxes.conf[i].cpu().numpy())
                                
                                detection = Detection(
                                    class_id=class_id,
                                    class_name=class_name,
                                    confidence=round(confidence, 4),
                                    bbox=(
                                        float(xyxyn[0]),
                                        float(xyxyn[1]),
                                        float(xyxyn[2]),
                                        float(xyxyn[3]),
                                    ),
                                )
                                detections.append(detection)
                                
                                # Count unique objects
                                object_counts[class_name] = object_counts.get(class_name, 0) + 1
                        
                        frame_results.append(FrameDetections(
                            frame_number=idx,
                            timestamp_ms=timestamp_ms,
                            detections=detections,
                        ))
                        
                    except Exception as e:
                        logger.warning(f"Failed to process frame {idx}: {e}")
                        continue
                    
                    # Log progress every 100 frames
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(frame_paths)} frames")
                
                logger.info(
                    f"Visual analysis complete: {len(frame_results)} frames processed, "
                    f"{len(object_counts)} unique object types"
                )
                
        except Exception as e:
            logger.error(f"Visual analysis failed for video {video_id}: {e}")
            raise ProcessingError(
                message=f"Visual analysis failed: {str(e)}",
                stage="visual_intelligence",
                video_id=video_id,
                original_error=str(e),
            ) from e
        
        return VisualAnalysisResult(
            frames=frame_results,
            total_frames=len(frame_results),
            unique_objects=object_counts,
            version_hash=self._version_hash,
        )
    
    def get_object_timeline(
        self,
        result: VisualAnalysisResult,
        object_name: str,
        confidence_threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Get timeline of when a specific object appears.
        
        Returns list of time ranges where object is detected.
        """
        appearances: list[dict[str, Any]] = []
        
        for frame in result.frames:
            for det in frame.detections:
                if det.class_name.lower() == object_name.lower() and det.confidence >= confidence_threshold:
                    appearances.append({
                        "timestamp_ms": frame.timestamp_ms,
                        "confidence": det.confidence,
                        "bbox": det.bbox,
                    })
        
        return appearances