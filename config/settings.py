"""Application settings and configuration."""

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="neuro-hub", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_reload: bool = Field(default=False, description="Auto-reload on code changes")

    # MCP Server
    mcp_server_name: str = Field(default="neuro-hub-memory", description="MCP server name")
    mcp_server_version: str = Field(default="1.0.0", description="MCP server version")

    # Vector Database (Milvus)
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    milvus_user: str = Field(default="", description="Milvus user")
    milvus_password: str = Field(default="", description="Milvus password")
    milvus_db_name: str = Field(default="neuro_hub", description="Milvus database name")
    milvus_collection_prefix: str = Field(default="nh_", description="Collection prefix")

    # Graph Database (Neo4j)
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j user")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database")

    # Cache (Redis)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: str = Field(default="", description="Redis password")
    redis_ttl: int = Field(default=3600, description="Redis TTL in seconds")
    redis_max_connections: int = Field(default=50, description="Redis max connections")

    # Relational Database (PostgreSQL)
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="neuro_hub", description="PostgreSQL user")
    postgres_password: str = Field(default="password", description="PostgreSQL password")
    postgres_db: str = Field(default="neuro_hub", description="PostgreSQL database")

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model"
    )
    embedding_dimension: int = Field(default=384, description="Embedding dimension")
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    embedding_device: str = Field(default="cpu", description="Device for embeddings")
    embedding_cache_size: int = Field(default=10000, description="Embedding cache size")

    # Multi-Modal Embeddings
    voyage_api_key: str = Field(default="", description="Voyage API key")
    amazon_nova_api_key: str = Field(default="", description="Amazon Nova API key")

    # Search & Retrieval
    hybrid_search_alpha: float = Field(
        default=0.5, description="Hybrid search weight (0=BM25, 1=Vector)"
    )
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    rerank_model: str = Field(default="", description="Reranking model")
    rerank_top_k: int = Field(default=100, description="Candidates for reranking")
    rerank_final_k: int = Field(default=10, description="Final results after reranking")

    # Memory Consolidation
    consolidation_enabled: bool = Field(default=True, description="Enable consolidation")
    consolidation_batch_size: int = Field(default=100, description="Consolidation batch size")
    consolidation_interval_seconds: int = Field(
        default=300, description="Consolidation interval"
    )
    entity_resolution_threshold: float = Field(
        default=0.85, description="Entity resolution threshold"
    )
    deduplication_threshold: float = Field(default=0.90, description="Deduplication threshold")

    # Authentication
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production", description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=15, description="Access token expiry"
    )
    jwt_refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")
    oauth2_enabled: bool = Field(default=False, description="Enable OAuth2")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Requests per hour")
    rate_limit_per_day: int = Field(default=10000, description="Requests per day")

    # Observability
    otel_enabled: bool = Field(default=True, description="Enable OpenTelemetry")
    otel_service_name: str = Field(default="neuro-hub", description="OTEL service name")
    otel_exporter_jaeger_endpoint: str = Field(
        default="http://localhost:14268/api/traces", description="Jaeger endpoint"
    )
    prometheus_port: int = Field(default=9090, description="Prometheus port")

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1", description="Celery broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2", description="Celery result backend"
    )

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"], description="CORS origins"
    )
    cors_allow_credentials: bool = Field(default=True, description="CORS allow credentials")
    cors_allow_methods: List[str] = Field(default=["*"], description="CORS allow methods")
    cors_allow_headers: List[str] = Field(default=["*"], description="CORS allow headers")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @property
    def database_url(self) -> str:
        """Get PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()
