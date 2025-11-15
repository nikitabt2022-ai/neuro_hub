"""Hybrid search combining BM25 and vector search with RRF fusion."""

from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class HybridSearch:
    """Hybrid search using BM25 + Vector search with Reciprocal Rank Fusion."""

    def __init__(self) -> None:
        """Initialize hybrid search."""
        self.alpha = settings.hybrid_search_alpha  # 0 = full BM25, 1 = full vector
        self.k1 = settings.bm25_k1
        self.b = settings.bm25_b

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.lower().split()

    def _bm25_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        limit: int = 100,
    ) -> List[Tuple[int, float]]:
        """Perform BM25 search.

        Args:
            query: Search query
            documents: List of documents with 'content' field
            limit: Maximum results

        Returns:
            List of (index, score) tuples
        """
        if not documents:
            return []

        # Tokenize documents
        tokenized_docs = [self._tokenize(doc["content"]) for doc in documents]

        # Create BM25 index
        bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get scores
        scores = bm25.get_scores(tokenized_query)

        # Sort by score and return top-k with indices
        scored_docs = [(i, float(score)) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:limit]

    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        """Combine results using Reciprocal Rank Fusion.

        Args:
            bm25_results: BM25 results as (index, score) tuples
            vector_results: Vector results as (index, score) tuples
            k: Constant for RRF (default 60)

        Returns:
            Fused results as (index, score) tuples
        """
        rrf_scores: Dict[int, float] = {}

        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + self.alpha / (k + rank)

        # Add vector scores
        for rank, (idx, _) in enumerate(vector_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1 - self.alpha) / (k + rank)

        # Sort by RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results

    def search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        vector_scores: List[Tuple[int, float]],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search.

        Args:
            query: Search query
            documents: List of documents
            vector_scores: Vector search results as (index, score) tuples
            limit: Maximum results

        Returns:
            Ranked list of documents
        """
        # Perform BM25 search
        bm25_results = self._bm25_search(query, documents, limit=100)

        # Fuse results
        fused_results = self._reciprocal_rank_fusion(bm25_results, vector_scores)

        # Get top results
        top_results = fused_results[:limit]

        # Build final result list
        ranked_docs = []
        for rank, (idx, score) in enumerate(top_results, start=1):
            if idx < len(documents):
                doc = documents[idx].copy()
                doc["hybrid_score"] = score
                doc["rank"] = rank
                ranked_docs.append(doc)

        logger.debug(
            "Hybrid search completed",
            query_length=len(query),
            results=len(ranked_docs),
        )

        return ranked_docs
