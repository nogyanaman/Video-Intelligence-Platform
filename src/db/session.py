"""
Async Database Session Manager.
Handles connection pooling and session lifecycle.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from src.core.config import settings
from src.db.models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages async database connections and sessions."""
    
    _instance: DatabaseManager | None = None
    
    def __new__(cls) -> DatabaseManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize database manager."""
        if self._initialized:
            return
        
        self._initialized = True
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
    
    async def init(self) -> None:
        """Initialize database engine and session factory."""
        if self._engine is not None:
            return
        
        logger.info("Initializing database connection")
        
        self._engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("Database connection initialized")
    
    async def create_tables(self) -> None:
        """Create all tables (for development/testing)."""
        if self._engine is None:
            await self.init()
        
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created")
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async session context manager."""
        if self._session_factory is None:
            await self.init()
        
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    def get_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get the session factory (for dependency injection)."""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._session_factory


# Singleton instance
db_manager = DatabaseManager()


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for getting async sessions."""
    async with db_manager.session() as session:
        yield session