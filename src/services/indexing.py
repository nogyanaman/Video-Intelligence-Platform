"""
Indexing Service - The "Brain"

Uses Sentence-Transformers (all-MiniLM-L6-v2) and CLIP for vectorization.
Stores vectors in Milvus for semantic search.

CRITICAL: Uses GPU Governor for VRAM safety.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

import numpy as np
from PIL import Image
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from src.core.config import settings
from src.core.exceptions import ProcessingError
from src.governor.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)


@dataclass
class TextEmbedding:
    """Text embedding with metadata."""
    text: str
    embedding: list[float]
    start_ms: int
    end_ms: int
    segment_index: int


@dataclass
class ImageEmbedding:
    """Image embedding with metadata."""
    frame_number: int
    timestamp_ms: int
    embedding: list[float]


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    text_embeddings_count: int
    image_embeddings_count: int
    version_hash: str


class IndexingService:
    """
    Vector indexing service for semantic search.
    
    Models:
    - Text: all-MiniLM-L6-v2 (Sentence-Transformers)
    - Image: CLIP (ViT-B/32)
    
    Vector Database: Milvus 2.4
    """
    
    TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    CLIP_MODEL = "ViT-B-32"
    TEXT_DIM = 384  # all-MiniLM-L6-v2 dimension
    IMAGE_DIM = 512  # CLIP ViT-B/32 dimension
    
    # Collection names
    TEXT_COLLECTION = "video_text_embeddings"
    IMAGE_COLLECTION = "video_image_embeddings"
    
    MODEL_VERSION = "indexing-v1"
    
    def __init__(self) -> None:
        """Initialize indexing service."""
        self._text_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._version_hash = self._compute_version_hash()
        self._milvus_connected = False
    
    def _compute_version_hash(self) -> str:
        """Compute version hash for caching."""
        version_string = f"{self.TEXT_MODEL}-{self.CLIP_MODEL}-{self.MODEL_VERSION}"
        return hashlib.sha256(version_string.encode()).hexdigest()[:16]
    
    @property
    def version_hash(self) -> str:
        """Get model version hash."""
        return self._version_hash
    
    def _connect_milvus(self) -> None:
        """Connect to Milvus."""
        if self._milvus_connected:
            return
        
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        self._milvus_connected = True
        logger.info("Connected to Milvus")
    
    def _ensure_collections(self) -> None:
        """Ensure Milvus collections exist."""
        self._connect_milvus()
        
        # Text embeddings collection
        if not utility.has_collection(self.TEXT_COLLECTION):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                    FieldSchema(name="start_ms", dtype=DataType.INT64),
                    FieldSchema(name="end_ms", dtype=DataType.INT64),
                    FieldSchema(name="segment_index", dtype=DataType.INT32),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.TEXT_DIM),
                ],
                description="Video transcript text embeddings",
            )
            collection = Collection(name=self.TEXT_COLLECTION, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info(f"Created collection: {self.TEXT_COLLECTION}")
        
        # Image embeddings collection
        if not utility.has_collection(self.IMAGE_COLLECTION):
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                    FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="frame_number", dtype=DataType.INT32),
                    FieldSchema(name="timestamp_ms", dtype=DataType.INT64),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.IMAGE_DIM),
                ],
                description="Video frame image embeddings",
            )
            collection = Collection(name=self.IMAGE_COLLECTION, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                },
            )
            logger.info(f"Created collection: {self.IMAGE_COLLECTION}")
    
    def _load_text_model(self) -> Any:
        """Load Sentence-Transformers model."""
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"Loading text model: {self.TEXT_MODEL}")
        
        model = SentenceTransformer(
            self.TEXT_MODEL,
            device="cuda",
            cache_folder=str(settings.models_dir),
        )
        
        logger.info("Text model loaded")
        return model
    
    def _unload_text_model(self, model: Any) -> None:
        """Unload text model."""
        logger.info("Unloading text model")
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    
    def _load_clip_model(self) -> tuple[Any, Any]:
        """Load CLIP model."""
        import open_clip
        import torch
        
        logger.info(f"Loading CLIP model: {self.CLIP_MODEL}")
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.CLIP_MODEL,
            pretrained="openai",
            cache_dir=str(settings.models_dir),
        )
        model = model.cuda()
        model.eval()
        
        logger.info("CLIP model loaded")
        return model, preprocess
    
    def _unload_clip_model(self, model_tuple: tuple[Any, Any]) -> None:
        """Unload CLIP model."""
        logger.info("Unloading CLIP model")
        model, _ = model_tuple
        try:
            model.cpu()
        except Exception:
            pass
        del model
    
    def index_text(
        self,
        video_id: str,
        segments: list[dict[str, Any]],
    ) -> int:
        """
        Index transcript segments as text embeddings.
        
        Args:
            video_id: Video ID
            segments: List of transcript segments with start_ms, end_ms, text
            
        Returns:
            Number of indexed segments
        """
        if not segments:
            logger.info(f"No text segments to index for video {video_id}")
            return 0
        
        logger.info(f"Indexing {len(segments)} text segments for video {video_id}")
        
        self._ensure_collections()
        
        embeddings_data: list[dict[str, Any]] = []
        
        try:
            # Use GPU Governor for text model
            with gpu_manager.model_context(
                model_name="sentence-transformers",
                loader=self._load_text_model,
                unloader=self._unload_text_model,
            ) as model:
                
                # Extract texts for batch encoding
                texts = [seg.get("text", "") for seg in segments]
                
                # Generate embeddings in batches
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_embeddings = model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                
                # Prepare data for Milvus
                for idx, (seg, embedding) in enumerate(zip(segments, all_embeddings)):
                    embeddings_data.append({
                        "id": str(uuid.uuid4()),
                        "video_id": video_id,
                        "text": seg.get("text", "")[:4000],  # Truncate to max length
                        "start_ms": seg.get("start_ms", 0),
                        "end_ms": seg.get("end_ms", 0),
                        "segment_index": idx,
                        "embedding": embedding,
                    })
            
            # Insert into Milvus (outside GPU context)
            collection = Collection(self.TEXT_COLLECTION)
            collection.insert(embeddings_data)
            collection.flush()
            
            logger.info(f"Indexed {len(embeddings_data)} text embeddings")
            return len(embeddings_data)
            
        except Exception as e:
            logger.error(f"Text indexing failed for video {video_id}: {e}")
            raise ProcessingError(
                message=f"Text indexing failed: {str(e)}",
                stage="indexing",
                video_id=video_id,
                original_error=str(e),
            ) from e
    
    def index_images(
        self,
        video_id: str,
        frame_paths: list[Path],
        fps: int = 1,
    ) -> int:
        """
        Index keyframes as image embeddings using CLIP.
        
        Args:
            video_id: Video ID
            frame_paths: List of paths to frame images
            fps: Frames per second for timestamp calculation
            
        Returns:
            Number of indexed frames
        """
        if not frame_paths:
            logger.info(f"No frames to index for video {video_id}")
            return 0
        
        logger.info(f"Indexing {len(frame_paths)} frames for video {video_id}")
        
        self._ensure_collections()
        
        embeddings_data: list[dict[str, Any]] = []
        
        try:
            # Use GPU Governor for CLIP model
            def load_clip():
                return self._load_clip_model()
            
            with gpu_manager.model_context(
                model_name="clip",
                loader=load_clip,
                unloader=self._unload_clip_model,
            ) as model_tuple:
                import torch
                
                model, preprocess = model_tuple
                
                # Process frames sequentially
                for idx, frame_path in enumerate(frame_paths):
                    if not frame_path.exists():
                        continue
                    
                    try:
                        # Load and preprocess image
                        image = Image.open(frame_path).convert("RGB")
                        image_tensor = preprocess(image).unsqueeze(0).cuda()
                        
                        # Generate embedding
                        with torch.no_grad():
                            embedding = model.encode_image(image_tensor)
                            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                            embedding = embedding.cpu().numpy().flatten().tolist()
                        
                        # Calculate timestamp
                        timestamp_ms = int((idx / fps) * 1000)
                        
                        embeddings_data.append({
                            "id": str(uuid.uuid4()),
                            "video_id": video_id,
                            "frame_number": idx,
                            "timestamp_ms": timestamp_ms,
                            "embedding": embedding,
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to process frame {idx}: {e}")
                        continue
                    
                    # Log progress
                    if (idx + 1) % 100 == 0:
                        logger.info(f"Processed {idx + 1}/{len(frame_paths)} frames for indexing")
            
            # Insert into Milvus (outside GPU context)
            if embeddings_data:
                collection = Collection(self.IMAGE_COLLECTION)
                collection.insert(embeddings_data)
                collection.flush()
            
            logger.info(f"Indexed {len(embeddings_data)} image embeddings")
            return len(embeddings_data)
            
        except Exception as e:
            logger.error(f"Image indexing failed for video {video_id}: {e}")
            raise ProcessingError(
                message=f"Image indexing failed: {str(e)}",
                stage="indexing",
                video_id=video_id,
                original_error=str(e),
            ) from e
    
    def search_text(
        self,
        query: str,
        video_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search for similar text segments.
        
        Args:
            query: Search query
            video_id: Optional filter by video ID
            limit: Maximum results
            
        Returns:
            List of matching segments with scores
        """
        self._ensure_collections()
        
        try:
            # Load text model for query encoding
            with gpu_manager.model_context(
                model_name="sentence-transformers",
                loader=self._load_text_model,
                unloader=self._unload_text_model,
            ) as model:
                query_embedding = model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )[0].tolist()
            
            # Search in Milvus
            collection = Collection(self.TEXT_COLLECTION)
            collection.load()
            
            expr = f'video_id == "{video_id}"' if video_id else None
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                expr=expr,
                output_fields=["video_id", "text", "start_ms", "end_ms", "segment_index"],
            )
            
            matches = []
            for hits in results:
                for hit in hits:
                    matches.append({
                        "video_id": hit.entity.get("video_id"),
                        "text": hit.entity.get("text"),
                        "start_ms": hit.entity.get("start_ms"),
                        "end_ms": hit.entity.get("end_ms"),
                        "segment_index": hit.entity.get("segment_index"),
                        "score": float(hit.score),
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise ProcessingError(
                message=f"Text search failed: {str(e)}",
                stage="search",
            ) from e
    
    def delete_video_embeddings(self, video_id: str) -> None:
        """Delete all embeddings for a video."""
        self._ensure_collections()
        
        try:
            # Delete from text collection
            text_collection = Collection(self.TEXT_COLLECTION)
            text_collection.delete(expr=f'video_id == "{video_id}"')
            
            # Delete from image collection
            image_collection = Collection(self.IMAGE_COLLECTION)
            image_collection.delete(expr=f'video_id == "{video_id}"')
            
            logger.info(f"Deleted embeddings for video {video_id}")
            
        except Exception as e:
            logger.warning(f"Failed to delete embeddings for video {video_id}: {e}")