"""Test configuration module."""

import pytest
import os
from config.settings import Settings


class TestSettings:
    """Test settings configuration."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.app_name == "neuro-hub"
        assert settings.app_version == "1.0.0"
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_database_url_property(self):
        """Test database URL construction."""
        settings = Settings(
            postgres_user="test_user",
            postgres_password="test_pass",
            postgres_host="localhost",
            postgres_port=5432,
            postgres_db="test_db",
        )

        expected = "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
        assert settings.database_url == expected

    def test_redis_url_property(self):
        """Test Redis URL construction."""
        # Without password
        settings = Settings(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password="",
        )
        assert settings.redis_url == "redis://localhost:6379/0"

        # With password
        settings_with_pass = Settings(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password="secret",
        )
        assert settings_with_pass.redis_url == "redis://:secret@localhost:6379/0"

    def test_log_level_validation(self):
        """Test log level validation."""
        # Valid log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Invalid log level
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_embedding_settings(self):
        """Test embedding configuration."""
        settings = Settings()

        assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.embedding_dimension == 384
        assert settings.embedding_batch_size == 32
        assert settings.embedding_device == "cpu"

    def test_memory_consolidation_settings(self):
        """Test consolidation configuration."""
        settings = Settings()

        assert settings.consolidation_enabled is True
        assert settings.entity_resolution_threshold == 0.85
        assert settings.deduplication_threshold == 0.90
