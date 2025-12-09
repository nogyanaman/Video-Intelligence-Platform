"""
Intelligence Router - API Query Layer

Handles user queries using:
1. Intent Parsing via Ollama (Llama 3)
2. Vector Search in Milvus
3. RAG Generation for final answers

Includes Redis caching for optimization.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.core.config import settings
from src.core.exceptions import OllamaError, ProcessingError
from src.services.indexing import IndexingService
from src.storage.redis_client import redis_client

logger = logging.getLogger(__name__)


@dataclass
class ParsedIntent:
    """Parsed query intent from LLM."""
    search_terms: list[str]
    required_modules: list[str]
    query_type: str
    original_query: str


@dataclass
class QueryResult:
    """Final query result."""
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    cached: bool = False


class IntelligenceRouter:
    """
    Routes user queries through intent parsing, search, and RAG.
    
    Flow:
    1. Check Redis cache for hot queries
    2. Parse intent using Ollama (Llama 3)
    3. Search Milvus using extracted search terms
    4. Generate final answer using RAG with Ollama
    5. Cache result
    """
    
    INTENT_PROMPT_TEMPLATE = """Analyze this user query about a video. Extract the key information needed to search the video content.

Query: {query}

Return a JSON object with the following structure:
{{
    "search_terms": ["list", "of", "search", "terms"],
    "required_modules": ["transcript", "objects"],
    "query_type": "factual|temporal|descriptive|counting"
}}

Rules:
- search_terms: Key words/phrases to search for in transcript and visual content
- required_modules: Which analysis to use - "transcript" for speech, "objects" for visual detection
- query_type: 
  - "factual" for questions about what happens
  - "temporal" for when something happens
  - "descriptive" for descriptions of scenes
  - "counting" for how many of something

Respond ONLY with valid JSON, no other text."""

    RAG_PROMPT_TEMPLATE = """Based on the following context from a video, answer the user's question.

Context from video:
{context}

User Question: {query}

Instructions:
- Use ONLY the information provided in the context
- If the context doesn't contain enough information, say so
- Include specific timestamps when relevant (format as MM:SS)
- Be concise but complete

Answer:"""

    def __init__(self) -> None:
        """Initialize intelligence router."""
        self.indexing_service = IndexingService()
    
    async def _call_ollama(
        self,
        prompt: str,
        stream: bool = False,
    ) -> str:
        """
        Call Ollama API for LLM inference.
        
        Note: Ollama runs in a separate container, so this doesn't compete
        for GPU memory with PyTorch models (Interlock protocol).
        """
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{settings.ollama_host}/api/generate",
                    json={
                        "model": settings.ollama_model,
                        "prompt": prompt,
                        "stream": stream,
                        "options": {
                            "temperature": 0.1,
                            "num_predict": 1024,
                        },
                    },
                )
                
                if response.status_code != 200:
                    raise OllamaError(
                        message=f"Ollama returned status {response.status_code}",
                        model=settings.ollama_model,
                    )
                
                result = response.json()
                return result.get("response", "")
                
        except httpx.TimeoutException:
            raise OllamaError(
                message="Ollama request timeout",
                model=settings.ollama_model,
            )
        except Exception as e:
            raise OllamaError(
                message=f"Ollama request failed: {str(e)}",
                model=settings.ollama_model,
            ) from e
    
    async def parse_intent(self, query: str) -> ParsedIntent:
        """
        Parse user query intent using Ollama.
        
        Returns structured intent with search terms and required modules.
        """
        logger.info(f"Parsing intent for query: {query[:100]}...")
        
        prompt = self.INTENT_PROMPT_TEMPLATE.format(query=query)
        response = await self._call_ollama(prompt)
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            return ParsedIntent(
                search_terms=parsed.get("search_terms", [query]),
                required_modules=parsed.get("required_modules", ["transcript"]),
                query_type=parsed.get("query_type", "factual"),
                original_query=query,
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse intent JSON: {e}. Using fallback.")
            return ParsedIntent(
                search_terms=[query],
                required_modules=["transcript", "objects"],
                query_type="factual",
                original_query=query,
            )
    
    async def search(
        self,
        intent: ParsedIntent,
        video_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search Milvus using parsed intent.
        
        Combines text search across all search terms.
        """
        all_results: list[dict[str, Any]] = []
        seen_segments: set[tuple[str, int]] = set()
        
        for term in intent.search_terms:
            try:
                results = self.indexing_service.search_text(
                    query=term,
                    video_id=video_id,
                    limit=limit // len(intent.search_terms) + 1,
                )
                
                for result in results:
                    key = (result["video_id"], result["start_ms"])
                    if key not in seen_segments:
                        seen_segments.add(key)
                        all_results.append(result)
                        
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {e}")
        
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:limit]
    
    def _format_timestamp(self, ms: int) -> str:
        """Format milliseconds as MM:SS."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def _build_context(self, search_results: list[dict[str, Any]]) -> str:
        """Build context string from search results."""
        context_parts = []
        
        for result in search_results:
            start_ts = self._format_timestamp(result.get("start_ms", 0))
            end_ts = self._format_timestamp(result.get("end_ms", 0))
            text = result.get("text", "")
            
            context_parts.append(f"[{start_ts} - {end_ts}]: {text}")
        
        return "\n".join(context_parts)
    
    async def generate_answer(
        self,
        query: str,
        search_results: list[dict[str, Any]],
    ) -> str:
        """
        Generate final answer using RAG with Ollama.
        """
        if not search_results:
            return "I couldn't find relevant information in the video to answer your question."
        
        context = self._build_context(search_results)
        prompt = self.RAG_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )
        
        answer = await self._call_ollama(prompt)
        return answer.strip()
    
    async def query(
        self,
        query: str,
        video_id: str | None = None,
    ) -> QueryResult:
        """
        Process a user query end-to-end.
        
        1. Check cache
        2. Parse intent
        3. Search
        4. Generate answer
        5. Cache result
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Check Redis cache (FIXED: includes video_id)
        try:
            await redis_client.init()
            cached = await redis_client.get_query_result(query, video_id)
            if cached:
                logger.info("Cache hit for query")
                return QueryResult(
                    answer=cached.get("answer", ""),
                    sources=cached.get("sources", []),
                    confidence=cached.get("confidence", 0.0),
                    cached=True,
                )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        # Step 2: Parse intent
        intent = await self.parse_intent(query)
        logger.info(f"Parsed intent: terms={intent.search_terms}, modules={intent.required_modules}")
        
        # Step 3: Search
        search_results = await self.search(intent, video_id)
        logger.info(f"Found {len(search_results)} search results")
        
        # Step 4: Generate answer
        answer = await self.generate_answer(query, search_results)
        
        # Calculate confidence based on search scores
        confidence = 0.0
        if search_results:
            avg_score = sum(r.get("score", 0) for r in search_results) / len(search_results)
            confidence = min(avg_score, 1.0)
        
        result = QueryResult(
            answer=answer,
            sources=[
                {
                    "video_id": r.get("video_id"),
                    "text": r.get("text"),
                    "start_ms": r.get("start_ms"),
                    "end_ms": r.get("end_ms"),
                    "score": r.get("score"),
                }
                for r in search_results
            ],
            confidence=round(confidence, 4),
            cached=False,
        )
        
        # Step 5: Cache result (FIXED: includes video_id)
        try:
            await redis_client.set_query_result(
                query,
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence,
                },
                video_id=video_id,
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
        
        return result