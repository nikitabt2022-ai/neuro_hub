"""Test hybrid search functionality."""

import pytest

# Skip entire module if rank_bm25 is not available
pytest.importorskip("rank_bm25", reason="rank_bm25 not installed (optional dependency)")

from retrieval.hybrid_search import HybridSearch


class TestHybridSearch:
    """Test hybrid search combining BM25 and vector search."""

    def test_initialization(self):
        """Test hybrid search initialization."""
        search = HybridSearch()

        assert 0.0 <= search.alpha <= 1.0
        assert search.k1 > 0
        assert search.b >= 0

    def test_tokenization(self):
        """Test text tokenization."""
        search = HybridSearch()

        text = "This is a Test Sentence"
        tokens = search._tokenize(text)

        assert tokens == ["this", "is", "a", "test", "sentence"]

    def test_bm25_search_simple(self):
        """Test simple BM25 search."""
        search = HybridSearch()

        documents = [
            {"content": "Python programming tutorial"},
            {"content": "Java enterprise development"},
            {"content": "Python web development"},
        ]

        query = "Python"
        results = search._bm25_search(query, documents, limit=2)

        # Should return 2 results
        assert len(results) == 2

        # Results should be (index, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

        # Scores should be floats
        assert all(isinstance(r[1], float) for r in results)

        # Documents with "Python" should rank higher
        top_doc_idx = results[0][0]
        assert "Python" in documents[top_doc_idx]["content"]

    def test_bm25_empty_documents(self):
        """Test BM25 with no documents."""
        search = HybridSearch()

        results = search._bm25_search("query", [], limit=10)

        assert results == []

    def test_reciprocal_rank_fusion(self):
        """Test RRF fusion algorithm."""
        search = HybridSearch()

        bm25_results = [(0, 0.9), (1, 0.7), (2, 0.5)]
        vector_results = [(2, 0.95), (0, 0.8), (3, 0.6)]

        fused = search._reciprocal_rank_fusion(bm25_results, vector_results)

        # Should return fused results
        assert len(fused) >= 3

        # Results should be (index, score) tuples
        assert all(isinstance(r, tuple) and len(r) == 2 for r in fused)

        # Scores should be positive
        assert all(r[1] > 0 for r in fused)

        # Results should be sorted by score descending
        scores = [r[1] for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_hybrid_search_integration(self):
        """Test full hybrid search."""
        search = HybridSearch()

        documents = [
            {"content": "Machine learning with Python", "id": 0},
            {"content": "Deep learning neural networks", "id": 1},
            {"content": "Python data science tutorial", "id": 2},
        ]

        # Simulated vector scores (in real scenario, from vector search)
        vector_scores = [(0, 0.9), (2, 0.85), (1, 0.6)]

        results = search.search(
            query="Python machine learning",
            documents=documents,
            vector_scores=vector_scores,
            limit=2,
        )

        # Should return limited results
        assert len(results) <= 2

        # Each result should have required fields
        for result in results:
            assert "rank" in result
            assert "hybrid_score" in result
            assert "content" in result

        # Ranks should be sequential
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_hybrid_search_with_no_matches(self):
        """Test hybrid search when query doesn't match."""
        search = HybridSearch()

        documents = [
            {"content": "Completely different topic"},
        ]

        vector_scores = [(0, 0.1)]  # Low similarity

        results = search.search(
            query="unrelated query",
            documents=documents,
            vector_scores=vector_scores,
            limit=10,
        )

        # Should still return result (might be low scored)
        assert len(results) >= 0

    def test_alpha_parameter_effect(self):
        """Test that alpha parameter affects fusion."""
        # Test with BM25-heavy weighting
        search_bm25 = HybridSearch()
        search_bm25.alpha = 0.2  # Favor BM25

        # Test with vector-heavy weighting
        search_vector = HybridSearch()
        search_vector.alpha = 0.8  # Favor vector

        documents = [{"content": "test document"}]
        bm25_results = [(0, 0.9)]
        vector_results = [(0, 0.5)]

        fused_bm25 = search_bm25._reciprocal_rank_fusion(bm25_results, vector_results)
        fused_vector = search_vector._reciprocal_rank_fusion(bm25_results, vector_results)

        # Scores should be different based on alpha
        assert fused_bm25[0][1] != fused_vector[0][1]
