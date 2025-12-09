"""
Health check and system status endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter

from src import __version__
from src.api.schemas import GPUStatusResponse, HealthResponse
from src.governor.gpu_manager import gpu_manager

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns system status including GPU information.
    """
    status = gpu_manager.get_vram_status()
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        gpu_available=gpu_manager._cuda_available,
        vram_used_gb=status.used_gb,
        vram_total_gb=status.total_gb,
        vram_state=status.state.value,
    )


@router.get("/gpu", response_model=GPUStatusResponse)
async def gpu_status() -> GPUStatusResponse:
    """
    Detailed GPU status endpoint.
    
    Returns comprehensive VRAM and GPU information.
    """
    status = gpu_manager.get_vram_status()
    
    return GPUStatusResponse(
        used_gb=status.used_gb,
        total_gb=status.total_gb,
        free_gb=status.free_gb,
        state=status.state.value,
        utilization_percent=status.utilization_percent,
        current_model=gpu_manager.get_current_model(),
        is_locked=gpu_manager.is_gpu_locked(),
    )


@router.post("/gpu/cleanup")
async def force_gpu_cleanup() -> dict:
    """
    Force GPU cleanup.
    
    Triggers aggressive garbage collection and CUDA cache clearing.
    """
    gpu_manager.force_cleanup()
    status = gpu_manager.get_vram_status()
    
    return {
        "message": "GPU cleanup completed",
        "vram_used_gb": status.used_gb,
        "vram_state": status.state.value,
    }