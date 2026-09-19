"""Centralised configuration constants for the pipeline.

All magic strings and default URLs live here so they can be changed
in one place (or overridden via environment variables).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Server URLs
# ---------------------------------------------------------------------------
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:8080")
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8081")

# ---------------------------------------------------------------------------
# Model identifiers (as served by llama-server)
# ---------------------------------------------------------------------------
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "qwen-embedding")

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
DATA_DIR: str = os.getenv("DATA_DIR", "data")
