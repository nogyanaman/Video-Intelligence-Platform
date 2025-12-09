"""Celery worker tasks."""
from src.workers.tasks import celery_app

__all__ = ["celery_app"]