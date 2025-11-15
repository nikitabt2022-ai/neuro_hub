"""Vector store implementation using Milvus."""

import json
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from config.settings import settings
from config.logging import StructuredLogger
from models.memory import Memory, MemoryType

logger = StructuredLogger(__name__)


class VectorStore:
    """Vector store for embeddings using Milvus."""

    def __init__(self) -> None:
        """Initialize vector store."""
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.user = settings.milvus_user
        self.password = settings.milvus_password
        self.db_name = settings.milvus_db_name
        self.collection_prefix = settings.milvus_collection_prefix
        self.dimension = settings.embedding_dimension
        self._connected = False

    def connect(self) -> None:
        """Connect to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=str(self.port),
                user=self.user,
                password=self.password,
                db_name=self.db_name,
            )
            self._connected = True
            logger.info("Connected to Milvus", host=self.host, port=self.port)
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._connected:
            connections.disconnect(alias="default")
            self._connected = False
            logger.info("Disconnected from Milvus")

    def _get_collection_name(self, collection: str, memory_type: MemoryType) -> str:
        """Get full collection name."""
        return f"{self.collection_prefix}{collection}_{memory_type.value}"

    def _create_collection_schema(self) -> CollectionSchema:
        """Create collection schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
        ]
        return CollectionSchema(fields=fields, description="Memory collection")

    def create_collection(self, collection: str, memory_type: MemoryType) -> None:
        """Create a collection if it doesn't exist."""
        collection_name = self._get_collection_name(collection, memory_type)

        if utility.has_collection(collection_name):
            logger.debug(f"Collection already exists: {collection_name}")
            return

        schema = self._create_collection_schema()
        col = Collection(name=collection_name, schema=schema)

        # Create index for vector field
        index_params = {
            "metric_type": "IP",  # Inner Product (cosine similarity)
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        }
        col.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection: {collection_name}")

    def insert(
        self,
        memory: Memory,
    ) -> None:
        """Insert memory into vector store."""
        if not memory.embedding:
            raise ValueError("Memory must have an embedding")

        collection_name = self._get_collection_name(memory.collection, memory.type)

        # Ensure collection exists
        self.create_collection(memory.collection, memory.type)

        # Prepare data
        data = [
            [str(memory.id)],
            [memory.content],
            [memory.embedding],
            [memory.metadata.model_dump()],
            [int(memory.created_at.timestamp() * 1000)],
            [int(memory.updated_at.timestamp() * 1000)],
        ]

        # Insert
        col = Collection(collection_name)
        col.insert(data)
        col.flush()
        logger.debug(f"Inserted memory {memory.id} into {collection_name}")

    def search(
        self,
        embedding: List[float],
        collection: str,
        memory_type: MemoryType,
        limit: int = 10,
        min_similarity: float = 0.0,
        filters: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar memories."""
        collection_name = self._get_collection_name(collection, memory_type)

        if not utility.has_collection(collection_name):
            return []

        col = Collection(collection_name)
        col.load()

        search_params = {
            "metric_type": "IP",
            "params": {"ef": 128},
        }

        results = col.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=filters,
            output_fields=["id", "content", "metadata", "created_at", "updated_at"],
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                # Filter by minimum similarity
                if hit.score < min_similarity:
                    continue

                formatted_results.append(
                    {
                        "id": hit.entity.get("id"),
                        "content": hit.entity.get("content"),
                        "metadata": hit.entity.get("metadata"),
                        "similarity_score": float(hit.score),
                        "created_at": hit.entity.get("created_at"),
                        "updated_at": hit.entity.get("updated_at"),
                    }
                )

        return formatted_results

    def get(self, memory_id: UUID, collection: str, memory_type: MemoryType) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        collection_name = self._get_collection_name(collection, memory_type)

        if not utility.has_collection(collection_name):
            return None

        col = Collection(collection_name)
        col.load()

        expr = f'id == "{str(memory_id)}"'
        results = col.query(
            expr=expr,
            output_fields=["id", "content", "metadata", "created_at", "updated_at"],
        )

        if not results:
            return None

        result = results[0]
        return {
            "id": result["id"],
            "content": result["content"],
            "metadata": result["metadata"],
            "created_at": result["created_at"],
            "updated_at": result["updated_at"],
        }

    def delete(self, memory_id: UUID, collection: str, memory_type: MemoryType) -> bool:
        """Delete a memory."""
        collection_name = self._get_collection_name(collection, memory_type)

        if not utility.has_collection(collection_name):
            return False

        col = Collection(collection_name)
        expr = f'id == "{str(memory_id)}"'
        col.delete(expr)
        col.flush()
        logger.debug(f"Deleted memory {memory_id} from {collection_name}")
        return True

    def list_collections(self) -> List[str]:
        """List all collections."""
        all_collections = utility.list_collections()
        # Filter by prefix and extract base names
        filtered = []
        for col in all_collections:
            if col.startswith(self.collection_prefix):
                # Remove prefix
                name = col[len(self.collection_prefix):]
                filtered.append(name)
        return filtered

    def get_collection_stats(
        self, collection: str, memory_type: MemoryType
    ) -> Dict[str, Any]:
        """Get collection statistics."""
        collection_name = self._get_collection_name(collection, memory_type)

        if not utility.has_collection(collection_name):
            return {"count": 0, "exists": False}

        col = Collection(collection_name)
        col.load()

        stats = col.num_entities
        return {
            "count": stats,
            "exists": True,
            "name": collection_name,
        }
