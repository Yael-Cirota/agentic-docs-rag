"""
pipeline.py – Full indexing pipeline: chunk → embed → store in Pinecone.

Usage
-----
    from pipeline import build_index, load_index

    # First run: parse, embed, and push everything to Pinecone
    index = build_index(project_roots=["path/to/my-project"])

    # Subsequent runs: just connect to the existing Pinecone index
    index = load_index()
"""
from __future__ import annotations

import logging
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

import config
from loader import load_documents

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pinecone helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_pinecone_index(pc: Pinecone) -> any:
    """Return the Pinecone index, creating it if it does not exist."""
    existing = [idx.name for idx in pc.list_indexes()]

    if config.PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone index '%s' (%d dims, cosine)...",
            config.PINECONE_INDEX_NAME,
            config.EMBEDDING_DIMENSION,
        )
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=config.PINECONE_CLOUD,
                region=config.PINECONE_REGION,
            ),
        )
        logger.info("Pinecone index created.")
    else:
        logger.info("Pinecone index '%s' already exists.", config.PINECONE_INDEX_NAME)

    return pc.Index(config.PINECONE_INDEX_NAME)


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
      4. Upsert into Pinecone (with metadata).
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

    logger.info("=== Step 2/4: Connecting to Pinecone ===")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = _get_or_create_pinecone_index(pc)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
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
    logger.info("Ingested %d node(s) into Pinecone.", len(nodes))

    logger.info("=== Step 4/4: Building VectorStoreIndex ===")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=_build_query_embed_model(),
    )

    logger.info("Index ready. Pipeline complete.")
    return index


def load_index() -> VectorStoreIndex:
    """
    Connect to an *existing* Pinecone index and wrap it in a VectorStoreIndex.
    Use this on subsequent runs when the data is already indexed.
    """
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    pinecone_index = pc.Index(config.PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=_build_query_embed_model(),
    )
    logger.info("Loaded existing Pinecone index '%s'.", config.PINECONE_INDEX_NAME)
    return index
