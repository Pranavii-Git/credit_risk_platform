"""
Docker and data path utilities.
Handles differences between local dev and Docker container environments.
"""

import os
from pathlib import Path


def is_running_in_docker() -> bool:
    """Detect whether the app is running inside a Docker container."""
    return Path("/.dockerenv").exists() or os.getenv("DOCKER_ENV", "") == "1"


def get_data_path(filename: str) -> Path:
    """
    Return the correct path to a data file.
    In Docker: /app/data/filename
    Locally: ./data/filename (or DATA_DIR env var)
    """
    from src.utils.config import DATA_DIR
    return Path(DATA_DIR) / filename


def get_model_path() -> Path:
    """Return correct model path for current environment."""
    from src.utils.config import MODEL_PATH
    return Path(MODEL_PATH)


def get_db_path() -> Path:
    """Return correct SQLite DB path for current environment."""
    from src.utils.config import SQLITE_DB
    return Path(SQLITE_DB)


def environment_info() -> dict:
    """Return a dict of environment diagnostics for debugging."""
    from src.utils.config import DATA_DIR, MODEL_PATH, SQLITE_DB, LLM_PROVIDER, LLM_MODEL

    return {
        "in_docker": is_running_in_docker(),
        "data_dir": str(DATA_DIR),
        "model_path": str(MODEL_PATH),
        "sqlite_db": str(SQLITE_DB),
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "python_version": os.popen("python --version").read().strip(),
    }
