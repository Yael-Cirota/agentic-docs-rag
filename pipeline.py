"""
pipeline.py – Full indexing pipeline: chunk → embed → store in FAISS (local).

FAISS stores vectors on disk in ./faiss_store/ – no external service needed.
Cohere is still used for embedding (requires internet for that API call only).

Usage
-----
    from pipeline import build_index, load_index

    # First run: parse, embed, and save to disk
    index = build_index(project_roots=["path/to/my-project"])

    # Subsequent runs: load from disk (no re-embedding needed)
    index = load_index()
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import config
from loader import load_documents

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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_index(project_roots: list[str | Path]) -> VectorStoreIndex:
    """
    Full pipeline:
      1. Load .md files from *project_roots*.
      2. Chunk with MarkdownNodeParser + SentenceSplitter.
      3. Embed with Cohere.
      4. Store vectors in a local FAISS index on disk.
      5. Return a LlamaIndex VectorStoreIndex ready for querying.
    """
    logger.info("=== Step 1/4: Loading documents ===")
    documents = load_documents(project_roots)
    if not documents:
        raise ValueError(
            "No markdown documents found. "
            "Check that project_roots contains .md files."
        )
    logger.info("Loaded %d document(s).", len(documents))

    logger.info("=== Step 2/4: Setting up FAISS vector store (local) ===")
    # IndexFlatIP = inner-product (cosine when vectors are normalised by Cohere)
    faiss_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("=== Step 3/4: Running ingestion pipeline (chunk + embed) ===")
    embed_model = _build_embed_model()

    ingestion = IngestionPipeline(
        transformations=[
            *_build_node_parsers(),
            embed_model,
        ],
        vector_store=vector_store,
    )
    nodes = ingestion.run(documents=documents, show_progress=True)
    logger.info("Ingested %d node(s) into FAISS.", len(nodes))

    logger.info("=== Step 4/4: Building VectorStoreIndex & persisting to disk ===")
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        embed_model=_build_query_embed_model(),
    )

    FAISS_STORE_DIR.mkdir(exist_ok=True)
    storage_context.persist(persist_dir=str(FAISS_STORE_DIR))
    logger.info("Index saved to %s", FAISS_STORE_DIR)

    logger.info("Index ready. Pipeline complete.")
    return index


def load_index() -> VectorStoreIndex:
    """
    Load an existing FAISS index from disk.
    Use this on subsequent runs when the data is already indexed.
    """
    if not FAISS_STORE_DIR.exists():
        raise FileNotFoundError(
            f"No FAISS store found at {FAISS_STORE_DIR}. "
            "Run 'python index.py <project_root>' first."
        )

    logger.info("Loading FAISS index from %s ...", FAISS_STORE_DIR)
    vector_store = FaissVectorStore.from_persist_dir(str(FAISS_STORE_DIR))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=str(FAISS_STORE_DIR),
    )

    index = load_index_from_storage(
        storage_context=storage_context,
        embed_model=_build_query_embed_model(),
    )
    logger.info("FAISS index loaded (%d dims).", config.EMBEDDING_DIMENSION)
    return index
