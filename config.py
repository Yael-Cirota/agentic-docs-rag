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
COHERE_LLM_MODEL: str = os.getenv("COHERE_LLM_MODEL", "command-r-plus-08-2024")

# Cohere embed-v3 models produce 1024-dimensional vectors
EMBEDDING_DIMENSION: int = 1024

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ────────────────────────────────────────────────────────────────
# Number of chunks to retrieve from FAISS before postprocessing
TOP_K: int = int(os.getenv("TOP_K", "5"))
# Chunks with a similarity score below this threshold are silently dropped
SIMILARITY_CUTOFF: float = float(os.getenv("SIMILARITY_CUTOFF", "0.35"))

# ── Input validation ─────────────────────────────────────────────────────────
# Minimum characters a question must contain (after stripping whitespace)
MIN_QUESTION_LENGTH: int = int(os.getenv("MIN_QUESTION_LENGTH", "3"))
# Maximum characters accepted; longer questions are silently truncated
MAX_QUESTION_LENGTH: int = int(os.getenv("MAX_QUESTION_LENGTH", "500"))
# Nodes that survive SIMILARITY_CUTOFF but score below this trigger a
# low-confidence warning (distinct from "no results found")
LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.50"))

# ── Structured extraction store ───────────────────────────────────────────────
# Path to the JSON file that holds pre-extracted decisions, rules, warnings, etc.
STRUCTURED_STORE_PATH: str = "structured_store/items.json"

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
