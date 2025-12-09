"""
Custom Exception Classes for the Video Intelligence Platform.
All exceptions inherit from VIPException for consistent error handling.
"""
from __future__ import annotations

from typing import Any


class VIPException(Exception):
    """Base exception for Video Intelligence Platform."""
    
    def __init__(
        self,
        message: str,
        code: str = "VIP_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


class GPUMemoryError(VIPException):
    """Raised when GPU memory constraints are violated."""
    
    def __init__(
        self,
        message: str = "GPU memory limit exceeded",
        current_usage_gb: float | None = None,
        threshold_gb: float | None = None,
    ) -> None:
        details = {}
        if current_usage_gb is not None:
            details["current_usage_gb"] = current_usage_gb
        if threshold_gb is not None:
            details["threshold_gb"] = threshold_gb
        super().__init__(message=message, code="GPU_MEMORY_ERROR", details=details)


class ProcessingError(VIPException):
    """Raised when video processing fails."""
    
    def __init__(
        self,
        message: str,
        stage: str | None = None,
        video_id: str | None = None,
        original_error: str | None = None,
    ) -> None:
        details = {}
        if stage:
            details["stage"] = stage
        if video_id:
            details["video_id"] = video_id
        if original_error:
            details["original_error"] = original_error
        super().__init__(message=message, code="PROCESSING_ERROR", details=details)


class ValidationError(VIPException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message=message, code="VALIDATION_ERROR", details=details)


class StorageError(VIPException):
    """Raised when storage operations fail."""
    
    def __init__(
        self,
        message: str,
        bucket: str | None = None,
        object_name: str | None = None,
    ) -> None:
        details = {}
        if bucket:
            details["bucket"] = bucket
        if object_name:
            details["object_name"] = object_name
        super().__init__(message=message, code="STORAGE_ERROR", details=details)


class OllamaError(VIPException):
    """Raised when Ollama operations fail."""
    
    def __init__(
        self,
        message: str,
        model: str | None = None,
    ) -> None:
        details = {}
        if model:
            details["model"] = model
        super().__init__(message=message, code="OLLAMA_ERROR", details=details)


class ModelLoadError(VIPException):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        retry_count: int = 0,
    ) -> None:
        details = {"retry_count": retry_count}
        if model_name:
            details["model_name"] = model_name
        super().__init__(message=message, code="MODEL_LOAD_ERROR", details=details)


class LockAcquisitionError(VIPException):
    """Raised when GPU lock cannot be acquired."""
    
    def __init__(
        self,
        message: str = "Failed to acquire GPU lock",
        timeout: float | None = None,
    ) -> None:
        details = {}
        if timeout is not None:
            details["timeout"] = timeout
        super().__init__(message=message, code="LOCK_ACQUISITION_ERROR", details=details)


class InterlockViolationError(VIPException):
    """Raised when the VRAM interlock protocol is violated."""
    
    def __init__(
        self,
        message: str = "Interlock violation detected",
        active_process: str | None = None,
    ) -> None:
        details = {}
        if active_process:
            details["active_process"] = active_process
        super().__init__(message=message, code="INTERLOCK_VIOLATION", details=details)