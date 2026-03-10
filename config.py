"""
config.py – centralised configuration loaded from .env
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── SSL fix for Windows (Python 3.14 doesn't bundle CA certs) ──────────────────
try:
    import truststore
    truststore.inject_into_ssl()   # use Windows native certificate store
except ImportError:
    import certifi, ssl, os
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Copy .env.example to .env and fill in your credentials."
        )
    return value


# ── API credentials ──────────────────────────────────────────────────────────
COHERE_API_KEY: str = _require("COHERE_API_KEY")
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")  # optional – not used with FAISS

# ── Pinecone ─────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-docs")
PINECONE_CLOUD: str = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.getenv("PINECONE_REGION", "us-east-1")

# ── Models ───────────────────────────────────────────────────────────────────
# embed-multilingual-v3.0 handles Hebrew+English content well
COHERE_EMBED_MODEL: str = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
COHERE_LLM_MODEL: str = os.getenv("COHERE_LLM_MODEL", "command-r-plus")

# Cohere embed-v3 models produce 1024-dimensional vectors
EMBEDDING_DIMENSION: int = 1024

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Known Agentic Coding tool folder patterns ────────────────────────────────
#
#   Each entry maps a human-readable tool name to the list of directory
#   names / glob patterns that the tool typically uses inside a project repo.
#   Adjust these paths to match wherever your projects live on disk.
#
TOOL_FOLDER_MAP: dict[str, list[str]] = {
    "cursor": [".cursor", ".cursorrules", ".cursor/rules"],
    "claude_code": [".claude", "CLAUDE.md", ".claude/commands"],
    "kiro": [".kiro", ".kiro/specs", ".kiro/hooks", ".kiro/steering"],
    "copilot": [".github/copilot-instructions.md", ".github"],
    "cline": [".clinerules", ".cline"],
    "windsurf": [".windsurf", ".windsurfrules"],
}
