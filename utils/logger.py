"""
utils/logger.py
Centralized logging using loguru with file + console output.
"""

import sys
from loguru import logger
from config.settings import settings


def setup_logger():
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stdout,
        level=settings.log.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <white>{message}</white>",
        colorize=True,
    )

    # File handler
    logger.add(
        settings.log.log_file,
        level=settings.log.level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    return logger


# Initialize on import
setup_logger()

__all__ = ["logger"]
