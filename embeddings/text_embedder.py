"""Text embedding engine using sentence-transformers."""

from typing import List, Union
import torch
from sentence_transformers import SentenceTransformer
from config.settings import settings
from config.logging import StructuredLogger

logger = StructuredLogger(__name__)


class TextEmbedder:
    """Text embedding engine using sentence-transformers."""

    def __init__(self) -> None:
        """Initialize text embedder."""
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        self.device = settings.embedding_device
        self._model: SentenceTransformer | None = None

    def load_model(self) -> None:
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded successfully", dimension=self.dimension)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _get_model(self) -> SentenceTransformer:
        """Get the model instance."""
        if self._model is None:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Failed to load embedding model")
        return self._model

    def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text.

        Args:
            text: Single text or list of texts

        Returns:
            Embedding vector(s)
        """
        model = self._get_model()

        # Handle single text
        if isinstance(text, str):
            embedding = model.encode(
                text,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return embedding.tolist()

        # Handle batch
        embeddings = model.encode(
            text,
            batch_size=self.batch_size,
            convert_to_tensor=False,
            show_progress_bar=len(text) > 100,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query.

        Args:
            query: Search query text

        Returns:
            Query embedding vector
        """
        return self.embed(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        return self.embed(documents)

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score
        """
        model = self._get_model()

        # Convert to tensors
        tensor1 = torch.tensor(embedding1)
        tensor2 = torch.tensor(embedding2)

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            tensor1.unsqueeze(0), tensor2.unsqueeze(0)
        )

        return float(similarity.item())

    def get_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.dimension

    def batch_embed_with_metadata(
        self, texts: List[str], metadata: List[dict]
    ) -> List[dict]:
        """Embed texts and attach metadata.

        Args:
            texts: List of texts to embed
            metadata: List of metadata dicts (same length as texts)

        Returns:
            List of dicts with embeddings and metadata
        """
        embeddings = self.embed_documents(texts)

        results = []
        for i, embedding in enumerate(embeddings):
            results.append(
                {
                    "text": texts[i],
                    "embedding": embedding,
                    "metadata": metadata[i] if i < len(metadata) else {},
                }
            )

        return results
