"""Retrieval layer for Neuro Hub."""

# Optional import for HybridSearch (requires rank-bm25)
try:
    from retrieval.hybrid_search import HybridSearch
    __all__ = ["HybridSearch"]
except ImportError:
    __all__ = []
