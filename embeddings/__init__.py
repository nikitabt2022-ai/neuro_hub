"""Embedding engines for Neuro Hub."""

from embeddings.embedding_cache import EmbeddingCache

# Optional import for TextEmbedder (requires torch)
try:
    from embeddings.text_embedder import TextEmbedder
    __all__ = ["TextEmbedder", "EmbeddingCache"]
except ImportError:
    __all__ = ["EmbeddingCache"]
