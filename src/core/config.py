"""
Pydantic Settings Configuration.
Single source of truth for all application configuration.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and type hints."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application
    app_name: str = Field(default="video-intelligence-platform")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    secret_key: str = Field(default="change-me-in-production")
    
    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="vip_user")
    postgres_password: str = Field(default="vip_password")
    postgres_db: str = Field(default="video_intelligence")
    
    @computed_field
    @property
    def database_url(self) -> str:
        """Async PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @computed_field
    @property
    def database_url_sync(self) -> str:
        """Sync PostgreSQL connection URL for Alembic."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_password: str = Field(default="")
    
    @computed_field
    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"
    
    @computed_field
    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL."""
        return self.redis_url
    
    @computed_field
    @property
    def celery_result_backend(self) -> str:
        """Celery result backend URL."""
        return self.redis_url.replace("/0", "/1")
    
    # MinIO
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_secure: bool = Field(default=False)
    
    # Milvus
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    
    # Ollama
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3:8b")
    
    # GPU Settings
    gpu_lock_path: Path = Field(default=Path("/app/temp/gpu.lock"))
    vram_green_threshold_gb: float = Field(default=4.0)
    vram_yellow_threshold_gb: float = Field(default=5.0)
    vram_red_threshold_gb: float = Field(default=5.5)
    vram_safe_load_threshold_gb: float = Field(default=2.0)
    
    # Processing Limits
    max_video_size_mb: int = Field(default=500)
    max_video_duration_seconds: int = Field(default=600)  # 10 minutes
    processing_fps: int = Field(default=1)
    
    # Cache TTL (seconds)
    module_result_ttl: int = Field(default=86400)  # 24 hours
    hot_query_ttl: int = Field(default=300)  # 5 minutes
    
    # Paths
    temp_dir: Path = Field(default=Path("/app/temp"))
    models_dir: Path = Field(default=Path("/app/models"))
    
    @computed_field
    @property
    def max_video_size_bytes(self) -> int:
        """Maximum video size in bytes."""
        return self.max_video_size_mb * 1024 * 1024
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()