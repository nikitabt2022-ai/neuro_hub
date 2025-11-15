"""Logging configuration for Neuro Hub."""

import logging
import sys
from typing import Any, Dict
from config.settings import settings


def setup_logging() -> None:
    """Configure application logging."""
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Set third-party loggers to WARNING
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLogger:
    """Structured logger with consistent formatting."""

    def __init__(self, name: str):
        """Initialize structured logger.

        Args:
            name: Logger name
        """
        self.logger = get_logger(name)

    def _format_extra(self, extra: Dict[str, Any]) -> str:
        """Format extra data."""
        if not extra:
            return ""
        parts = [f"{k}={v}" for k, v in extra.items()]
        return f" | {' | '.join(parts)}"

    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message."""
        self.logger.debug(message + self._format_extra(extra))

    def info(self, message: str, **extra: Any) -> None:
        """Log info message."""
        self.logger.info(message + self._format_extra(extra))

    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message."""
        self.logger.warning(message + self._format_extra(extra))

    def error(self, message: str, **extra: Any) -> None:
        """Log error message."""
        self.logger.error(message + self._format_extra(extra))

    def critical(self, message: str, **extra: Any) -> None:
        """Log critical message."""
        self.logger.critical(message + self._format_extra(extra))
