"""
Redis Client for Caching.
Handles query result caching and module result caching.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as redis

from src.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client for caching."""
    
    _instance: RedisClient | None = None
    
    def __new__(cls) -> RedisClient:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize Redis client."""
        if self._initialized:
            return
        
        self._initialized = True
        self._client: redis.Redis | None = None
    
    async def init(self) -> None:
        """Initialize async Redis connection."""
        if self._client is not None:
            return
        
        self._client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.info("Redis client initialized")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Redis connection closed")
    
    def _get_query_key(self, query: str, video_id: str | None = None) -> str:
        """Generate cache key for query."""
        video_part = video_id or "all"
        query_hash = hash(query) & 0xFFFFFFFF
        return f"query:{video_part}:{query_hash}"
    
    def _get_module_key(
        self,
        video_id: str,
        module_name: str,
        version_hash: str,
    ) -> str:
        """Generate cache key for module result."""
        return f"module:{video_id}:{module_name}:{version_hash}"
    
    async def get_query_result(
        self,
        query: str,
        video_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get cached query result."""
        if self._client is None:
            await self.init()
        
        key = self._get_query_key(query, video_id)
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None
    
    async def set_query_result(
        self,
        query: str,
        result: dict[str, Any],
        video_id: str | None = None,
        ttl: int | None = None,
    ) -> None:
        """Cache query result."""
        if self._client is None:
            await self.init()
        
        key = self._get_query_key(query, video_id)
        ttl = ttl or settings.hot_query_ttl
        
        try:
            await self._client.setex(
                key,
                ttl,
                json.dumps(result),
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    
    async def get_module_result(
        self,
        video_id: str,
        module_name: str,
        version_hash: str,
    ) -> dict[str, Any] | None:
        """Get cached module result."""
        if self._client is None:
            await self.init()
        
        key = self._get_module_key(video_id, module_name, version_hash)
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
        return None
    
    async def set_module_result(
        self,
        video_id: str,
        module_name: str,
        version_hash: str,
        result: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Cache module result."""
        if self._client is None:
            await self.init()
        
        key = self._get_module_key(video_id, module_name, version_hash)
        ttl = ttl or settings.module_result_ttl
        
        try:
            await self._client.setex(
                key,
                ttl,
                json.dumps(result),
            )
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
    
    async def delete_video_cache(self, video_id: str) -> None:
        """Delete all cached data for a video."""
        if self._client is None:
            await self.init()
        
        try:
            pattern = f"module:{video_id}:*"
            async for key in self._client.scan_iter(match=pattern):
                await self._client.delete(key)
            logger.info(f"Deleted cache for video {video_id}")
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")


# Singleton instance
redis_client = RedisClient()