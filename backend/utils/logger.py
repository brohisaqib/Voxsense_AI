# ============================================================
# VoxSense - Logging Utility
# ============================================================

import sys
from pathlib import Path
from loguru import logger

from config.settings import settings


def setup_logger():
    """Configure loguru logger for the application."""
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
    )

    # File handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
    )

    logger.info(f"🚀 VoxSense logger initialized | Level: {settings.log_level}")
    return logger


# Initialize logger
setup_logger()

__all__ = ["logger"]
