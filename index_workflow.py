"""
index_workflow.py – LlamaIndex event-driven workflow for building or loading the FAISS index.

Event flow
----------
    StartEvent(reindex, project_roots)
        │
        ▼
    [check_index]
        │
        ├─ store on disk & reindex=False ──► IndexExistsEvent
        │                                         │
        │                                         ▼
        │                                   [load_from_disk] ──► StopEvent(index)
        │
        └─ store missing or reindex=True  ──► BuildIndexEvent
                                                  │
                                                  ▼
                                           [load_documents] ──► DocsLoadedEvent
                                                  │
                                                  ▼
                                           [chunk_nodes]    ──► NodesChunkedEvent
                                                  │
                                                  ▼
                                           [embed_and_store] ──► StopEvent(index)

Usage
-----
    import asyncio
    from index_workflow import run_indexing

    # Load from disk (default):
    index = asyncio.run(run_indexing())

    # Force a full rebuild:
    index = asyncio.run(run_indexing(reindex=True, project_roots=["./sample_project"]))
"""
from __future__ import annotations

import logging
from pathlib import Path

import faiss

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.vector_stores.faiss import FaissVectorStore

import config
from loader import load_documents
from pipeline import (
    FAISS_STORE_DIR,
    _build_embed_model,
    _build_node_parsers,
    _build_query_embed_model,
)

logger = logging.getLogger(__name__)

# Files that must all be present for a persisted FAISS store to be considered complete.
_REQUIRED_STORE_FILES: frozenset[str] = frozenset(
    {"default__vector_store.json", "docstore.json", "index_store.json"}
)


# ── Events ─────────────────────────────────────────────────────────────────────

class IndexExistsEvent(Event):
    """FAISS store already persisted on disk – take the fast load branch."""


class BuildIndexEvent(Event):
    """Store is missing or a full reindex was requested – take the build branch."""
    project_roots: list[str]


class DocsLoadedEvent(Event):
    """Markdown files have been discovered and loaded as LlamaIndex Documents."""
    documents: list   # list[Document]
    project_roots: list[str]


class NodesChunkedEvent(Event):
    """Documents have been parsed and split into node chunks."""
    nodes: list       # list[BaseNode]


# ── Workflow ───────────────────────────────────────────────────────────────────

class IndexingWorkflow(Workflow):
    """
    Decides at runtime whether to load an existing index from disk or rebuild
    it from scratch, then routes through the appropriate branch of the event
    graph.

    Each @step method has a single, typed input event and a typed output event,
    making the data contract between steps explicit.
    """

    # ── Decision gate ──────────────────────────────────────────────────────────

    @step
    async def check_index(self, ev: StartEvent) -> IndexExistsEvent | BuildIndexEvent:
        """
        Inspect the FAISS store directory and emit the appropriate routing event.

        Input : StartEvent  { reindex: bool, project_roots: list[str] }
        Output: IndexExistsEvent  –or–  BuildIndexEvent
        """
        reindex: bool       = ev.get("reindex", False)
        project_roots: list = ev.get("project_roots", [])

        store_ready = False
        if FAISS_STORE_DIR.exists():
            present = {f.name for f in FAISS_STORE_DIR.iterdir() if f.is_file()}
            missing = _REQUIRED_STORE_FILES - present
            if missing:
                logger.warning(
                    "[check_index] FAISS store is incomplete (missing: %s) "
                    "– forcing rebuild.",
                    missing,
                )
            else:
                store_ready = True

        if not reindex and store_ready:
            logger.info("[check_index] FAISS store is complete → load branch.")
            return IndexExistsEvent()

        logger.info(
            "[check_index] Store missing/incomplete or reindex=True → build branch "
            "(roots: %s).",
            project_roots,
        )
        return BuildIndexEvent(project_roots=project_roots)

    # ── Load branch ────────────────────────────────────────────────────────────

    @step
    async def load_from_disk(self, ev: IndexExistsEvent) -> StopEvent:
        """
        Load the persisted FAISS index without re-embedding anything.

        Input : IndexExistsEvent  (no payload – presence triggers the step)
        Output: StopEvent  { result: VectorStoreIndex }
        """
        logger.info("[load_from_disk] Loading from %s …", FAISS_STORE_DIR)
        vector_store = FaissVectorStore.from_persist_dir(str(FAISS_STORE_DIR))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(FAISS_STORE_DIR),
        )
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=_build_query_embed_model(),
        )
        logger.info("[load_from_disk] Done (%d dims).", config.EMBEDDING_DIMENSION)
        return StopEvent(result=index)

    # ── Build branch – step 1 / 3 ──────────────────────────────────────────────

    @step
    async def load_documents(self, ev: BuildIndexEvent) -> DocsLoadedEvent:
        """
        Discover and load all .md / .mdx files from the given project roots.

        Input : BuildIndexEvent  { project_roots: list[str] }
        Output: DocsLoadedEvent  { documents: list[Document], project_roots }
        """
        if not ev.project_roots:
            raise ValueError(
                "Cannot build index: no project roots provided. "
                "Pass --roots ./your-project or set PROJECT_ROOTS in app.py."
            )
        logger.info("[load_documents] Scanning roots: %s", ev.project_roots)
        documents = load_documents(ev.project_roots)
        if not documents:
            raise ValueError(
                "No markdown documents found. "
                "Check that project_roots contains .md files."
            )
        logger.info("[load_documents] Loaded %d document(s).", len(documents))
        return DocsLoadedEvent(documents=documents, project_roots=ev.project_roots)

    # ── Build branch – step 2 / 3 ──────────────────────────────────────────────

    @step
    async def chunk_nodes(self, ev: DocsLoadedEvent) -> NodesChunkedEvent:
        """
        Parse documents into chunks using MarkdownNodeParser → SentenceSplitter.

        Input : DocsLoadedEvent   { documents, project_roots }
        Output: NodesChunkedEvent { nodes: list[BaseNode] }
        """
        logger.info("[chunk_nodes] Chunking %d document(s) …", len(ev.documents))
        pipeline = IngestionPipeline(transformations=_build_node_parsers())
        nodes = pipeline.run(documents=ev.documents, show_progress=True)
        logger.info("[chunk_nodes] Produced %d node(s).", len(nodes))
        return NodesChunkedEvent(nodes=nodes)

    # ── Build branch – step 3 / 3 ──────────────────────────────────────────────

    @step
    async def embed_and_store(self, ev: NodesChunkedEvent) -> StopEvent:
        """
        Embed nodes with Cohere, store in FAISS (IndexFlatIP), and persist to disk.

        Input : NodesChunkedEvent { nodes: list[BaseNode] }
        Output: StopEvent         { result: VectorStoreIndex }
        """
        logger.info("[embed_and_store] Building FAISS vector store …")
        faiss_index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = _build_embed_model()
        index = VectorStoreIndex(
            nodes=ev.nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
        )

        FAISS_STORE_DIR.mkdir(exist_ok=True)
        storage_context.persist(persist_dir=str(FAISS_STORE_DIR))
        logger.info(
            "[embed_and_store] Saved to %s (%d nodes).",
            FAISS_STORE_DIR,
            len(ev.nodes),
        )
        return StopEvent(result=index)


# ── Public helper ───────────────────────────────────────────────────────────────

async def run_indexing(
    reindex: bool = False,
    project_roots: list[str] | None = None,
) -> VectorStoreIndex:
    """
    Async entry point: run the IndexingWorkflow and return the ready index.

    Parameters
    ----------
    reindex       : force a full rebuild even if a store is already on disk.
    project_roots : directories that contain .md files (required when reindex=True).
    """
    workflow = IndexingWorkflow(timeout=300)
    return await workflow.run(reindex=reindex, project_roots=project_roots or [])
