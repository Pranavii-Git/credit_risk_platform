"""
Centralised logging setup using loguru.
Import get_logger() in every module instead of using print().
"""

import sys
import os
from loguru import logger

_configured = False


def get_logger(name: str):
    """Return a named loguru logger, configured once per process."""
    global _configured
    if not _configured:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logger.remove()  # remove default handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                   "<level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="7 days",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
        )
        _configured = True
    return logger.bind(name=name)
