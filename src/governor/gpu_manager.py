"""
GPU Manager - The Hardware Governor (CRITICAL)

This is the CORE LOGIC module that enforces VRAM safety.
Implements the Mutex Governor pattern to prevent OOM errors on 6GB VRAM.

Rules Enforced:
1. Interlock: LLM (Ollama) and Vision/Audio models never load simultaneously
2. Strict Seriality: No parallel processing
3. Aggressive Cleanup: gc.collect() + torch.cuda.empty_cache() between stages
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import httpx
from filelock import FileLock, Timeout
from tenacity import retry, stop_after_attempt, wait_fixed

from src.core.config import settings
from src.core.exceptions import (
    GPUMemoryError,
    InterlockViolationError,
    LockAcquisitionError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VRAMState(str, Enum):
    """VRAM usage states with thresholds."""
    GREEN = "green"   # < 4GB: Safe to load models
    YELLOW = "yellow" # 4-5GB: Pause new jobs, allow current to finish
    RED = "red"       # > 5GB: CRITICAL - Force unload everything


@dataclass(frozen=True)
class VRAMStatus:
    """Current VRAM status information."""
    used_gb: float
    total_gb: float
    free_gb: float
    state: VRAMState
    utilization_percent: float


class GPUManager:
    """
    Singleton GPU Manager implementing the Mutex Governor pattern.
    
    This class is responsible for:
    1. Monitoring VRAM usage
    2. Enforcing the interlock protocol between Ollama and PyTorch models
    3. Managing GPU locks
    4. Aggressive cleanup between pipeline stages
    """
    
    _instance: GPUManager | None = None
    _lock = threading.Lock()
    
    def __new__(cls) -> GPUManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize GPU Manager (only once due to singleton)."""
        if self._initialized:
            return
            
        self._initialized = True
        self._model_lock = threading.Lock()
        self._current_model: str | None = None
        self._file_lock_path = settings.gpu_lock_path
        self._file_lock = FileLock(str(self._file_lock_path), timeout=300)
        
        # Thresholds in bytes
        self._green_threshold = settings.vram_green_threshold_gb * 1024**3
        self._yellow_threshold = settings.vram_yellow_threshold_gb * 1024**3
        self._red_threshold = settings.vram_red_threshold_gb * 1024**3
        self._safe_load_threshold = settings.vram_safe_load_threshold_gb * 1024**3
        
        # Check if CUDA is available
        self._cuda_available = self._check_cuda()
        
        logger.info(
            "GPU Manager initialized",
            extra={
                "cuda_available": self._cuda_available,
                "lock_path": str(self._file_lock_path),
            }
        )
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not installed, CUDA check failed")
            return False
    
    def get_vram_status(self) -> VRAMStatus:
        """Get current VRAM status using pynvml."""
        if not self._cuda_available:
            return VRAMStatus(
                used_gb=0.0,
                total_gb=6.0,
                free_gb=6.0,
                state=VRAMState.GREEN,
                utilization_percent=0.0,
            )
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            
            used_bytes = mem_info.used
            total_bytes = mem_info.total
            free_bytes = mem_info.free
            
        except Exception as e:
            logger.warning(f"Failed to get VRAM info via pynvml: {e}")
            # Fallback to torch
            try:
                import torch
                if torch.cuda.is_available():
                    used_bytes = torch.cuda.memory_allocated(0)
                    total_bytes = torch.cuda.get_device_properties(0).total_memory
                    free_bytes = total_bytes - used_bytes
                else:
                    raise RuntimeError("CUDA not available")
            except Exception as e2:
                logger.error(f"Failed to get VRAM info: {e2}")
                return VRAMStatus(
                    used_gb=0.0,
                    total_gb=6.0,
                    free_gb=6.0,
                    state=VRAMState.GREEN,
                    utilization_percent=0.0,
                )
        
        used_gb = used_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        utilization = (used_bytes / total_bytes) * 100
        
        # Determine state
        if used_bytes >= self._red_threshold:
            state = VRAMState.RED
        elif used_bytes >= self._yellow_threshold:
            state = VRAMState.YELLOW
        else:
            state = VRAMState.GREEN
        
        return VRAMStatus(
            used_gb=round(used_gb, 2),
            total_gb=round(total_gb, 2),
            free_gb=round(free_gb, 2),
            state=state,
            utilization_percent=round(utilization, 1),
        )
    
    def force_cleanup(self) -> None:
        """
        Aggressive cleanup: gc.collect() + torch.cuda.empty_cache().
        MUST be called between every pipeline stage.
        """
        logger.info("Performing aggressive GPU cleanup")
        
        # Python garbage collection
        gc.collect()
        gc.collect()  # Double collect for cyclic references
        
        if self._cuda_available:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset peak memory stats for monitoring
                    torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                logger.warning(f"CUDA cleanup failed: {e}")
        
        gc.collect()
        
        status = self.get_vram_status()
        logger.info(
            f"Cleanup complete. VRAM: {status.used_gb:.2f}GB/{status.total_gb:.2f}GB "
            f"({status.state.value})"
        )
    
    def check_vram_state(self) -> VRAMState:
        """Check current VRAM state and handle RED state."""
        status = self.get_vram_status()
        
        if status.state == VRAMState.RED:
            logger.critical(
                f"VRAM CRITICAL: {status.used_gb:.2f}GB used. "
                "Forcing emergency cleanup."
            )
            self.force_cleanup()
            self.unload_ollama_model()
            self.force_cleanup()
            
            # Re-check after cleanup
            status = self.get_vram_status()
            if status.state == VRAMState.RED:
                raise GPUMemoryError(
                    message="VRAM still critical after cleanup",
                    current_usage_gb=status.used_gb,
                    threshold_gb=settings.vram_red_threshold_gb,
                )
        
        return status.state
    
    def is_ollama_active(self) -> bool:
        """Check if Ollama has a model loaded."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{settings.ollama_host}/api/tags")
                if response.status_code == 200:
                    # Check if any model is currently loaded
                    ps_response = client.get(f"{settings.ollama_host}/api/ps")
                    if ps_response.status_code == 200:
                        data = ps_response.json()
                        models = data.get("models", [])
                        return len(models) > 0
        except Exception as e:
            logger.warning(f"Failed to check Ollama status: {e}")
        return False
    
    def unload_ollama_model(self) -> bool:
        """
        Force Ollama to unload its model immediately.
        Sends POST /api/generate with keep_alive: 0.
        """
        logger.info("Requesting Ollama to unload model")
        
        try:
            with httpx.Client(timeout=30.0) as client:
                # Send empty generate request with keep_alive: 0
                response = client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
                
                if response.status_code == 200:
                    logger.info("Ollama model unload requested successfully")
                    # Wait a moment for unload to complete
                    time.sleep(2)
                    return True
                else:
                    logger.warning(f"Ollama unload returned status {response.status_code}")
                    
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model: {e}")
        
        return False
    
    def wait_for_vram_below_threshold(
        self,
        threshold_gb: float | None = None,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """
        Wait until VRAM usage drops below threshold.
        
        Args:
            threshold_gb: Target VRAM threshold in GB (default: safe_load_threshold)
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds
            
        Returns:
            True if threshold was reached, False if timeout
        """
        if threshold_gb is None:
            threshold_gb = settings.vram_safe_load_threshold_gb
        
        threshold_bytes = threshold_gb * 1024**3
        start_time = time.time()
        
        logger.info(f"Waiting for VRAM to drop below {threshold_gb:.1f}GB")
        
        while time.time() - start_time < timeout:
            status = self.get_vram_status()
            
            if status.used_gb <= threshold_gb:
                logger.info(f"VRAM below threshold: {status.used_gb:.2f}GB")
                return True
            
            logger.debug(
                f"VRAM at {status.used_gb:.2f}GB, waiting for {threshold_gb:.1f}GB"
            )
            time.sleep(poll_interval)
        
        logger.warning(
            f"Timeout waiting for VRAM to drop. Current: {self.get_vram_status().used_gb:.2f}GB"
        )
        return False
    
    @contextmanager
    def acquire_gpu_lock(self, timeout: float = 300.0) -> Generator[None, None, None]:
        """
        Acquire GPU file lock for exclusive access.
        
        This implements the file-based mutex for GPU access.
        """
        logger.debug("Attempting to acquire GPU lock")
        
        try:
            self._file_lock.acquire(timeout=timeout)
            logger.info("GPU lock acquired")
            yield
        except Timeout:
            raise LockAcquisitionError(
                message=f"Failed to acquire GPU lock within {timeout}s",
                timeout=timeout,
            )
        finally:
            if self._file_lock.is_locked:
                self._file_lock.release()
                logger.info("GPU lock released")
    
    @contextmanager
    def model_context(
        self,
        model_name: str,
        loader: Callable[[], T],
        unloader: Callable[[T], None] | None = None,
    ) -> Generator[T, None, None]:
        """
        Context manager for safe model loading/unloading.
        
        Implements the full Interlock Protocol:
        1. Check if Ollama is active
        2. If yes, unload Ollama
        3. Wait for VRAM to drop below safe threshold
        4. Acquire file lock
        5. Load model -> yield -> unload -> cleanup -> release lock
        
        Args:
            model_name: Name of the model for logging
            loader: Callable that loads and returns the model
            unloader: Optional callable to unload the model
        """
        model = None
        
        try:
            with self._model_lock:
                # Step 1: Check VRAM state
                self.check_vram_state()
                
                # Step 2: Check if Ollama is active and unload
                if self.is_ollama_active():
                    logger.info("Ollama is active. Initiating interlock protocol.")
                    self.unload_ollama_model()
                    self.force_cleanup()
                
                # Step 3: Wait for VRAM to be safe
                if not self.wait_for_vram_below_threshold():
                    raise GPUMemoryError(
                        message="VRAM did not drop to safe level for model loading",
                        current_usage_gb=self.get_vram_status().used_gb,
                        threshold_gb=settings.vram_safe_load_threshold_gb,
                    )
                
                # Step 4: Acquire file lock
                with self.acquire_gpu_lock():
                    self._current_model = model_name
                    
                    # Step 5: Load model with retry
                    logger.info(f"Loading model: {model_name}")
                    model = self._load_with_retry(model_name, loader)
                    
                    try:
                        yield model
                    finally:
                        # Step 6: Unload model
                        logger.info(f"Unloading model: {model_name}")
                        if unloader and model is not None:
                            try:
                                unloader(model)
                            except Exception as e:
                                logger.warning(f"Model unloader failed: {e}")
                        
                        # Explicitly delete model reference
                        del model
                        model = None
                        
                        self._current_model = None
                        
                        # Step 7: Aggressive cleanup
                        self.force_cleanup()
                        
        except Exception as e:
            # Ensure cleanup on any error
            if model is not None:
                try:
                    if unloader:
                        unloader(model)
                except Exception:
                    pass
                del model
            
            self._current_model = None
            self.force_cleanup()
            raise
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(5),
        reraise=True,
    )
    def _load_with_retry(self, model_name: str, loader: Callable[[], T]) -> T:
        """
        Load model with retry logic.
        
        If loading fails, retry once after 5-second backoff.
        If it fails again, abort.
        """
        try:
            return loader()
        except Exception as e:
            logger.warning(f"Model load failed for {model_name}: {e}. Retrying...")
            self.force_cleanup()
            raise ModelLoadError(
                message=f"Failed to load model: {model_name}",
                model_name=model_name,
            ) from e
    
    def get_current_model(self) -> str | None:
        """Get the name of the currently loaded model."""
        return self._current_model
    
    def is_gpu_locked(self) -> bool:
        """Check if GPU is currently locked."""
        return self._file_lock.is_locked


# Singleton instance
gpu_manager = GPUManager()