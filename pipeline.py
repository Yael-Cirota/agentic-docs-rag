"""
pipeline.py – Shared helper factories: embedding models, node parsers, and the
FAISS store path.  The actual indexing logic lives in index_workflow.py.

NOTE
----
build_index() and load_index() have been moved to index_workflow.py.
Use ``from index_workflow import run_indexing`` instead.
"""
from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding

import config

logger = logging.getLogger(__name__)

# Directory where the FAISS index + docstore are persisted
FAISS_STORE_DIR = Path(__file__).parent / "faiss_store"


# ─────────────────────────────────────────────────────────────────────────────
# Embedding model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_embed_model() -> CohereEmbedding:
    return CohereEmbedding(
        api_key=config.COHERE_API_KEY,
        model_name=config.COHERE_EMBED_MODEL,
        input_type="search_document",   # used during indexing
    )


def _build_query_embed_model() -> CohereEmbedding:
    return CohereEmbedding(
        api_key=config.COHERE_API_KEY,
        model_name=config.COHERE_EMBED_MODEL,
        input_type="search_query",      # used during retrieval
    )


# ─────────────────────────────────────────────────────────────────────────────
# Node parsers
# ─────────────────────────────────────────────────────────────────────────────

def _build_node_parsers() -> list:
    """
    Two-stage chunking pipeline:
      1. MarkdownNodeParser  – splits on headings, preserving semantic sections.
      2. SentenceSplitter    – further splits large sections to CHUNK_SIZE tokens.

    This ensures we never embed an oversized chunk while keeping heading
    context intact wherever possible.
    """
    return [
        MarkdownNodeParser(),
        SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        ),
    ]


# (build_index / load_index removed – see index_workflow.py)
