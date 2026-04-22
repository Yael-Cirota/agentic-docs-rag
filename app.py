"""
app.py – Gradio UI for the Agentic Docs RAG system.

Run:
    python app.py

Optional flags:
    --index   Re-index all documents before launching (use after adding new files).

Architecture
------------
Every user query is handled by an event-driven LlamaIndex Workflow:

  ┌───────────────────────────────────────────┐
  │  IndexingWorkflow                          │
  │  StartEvent → check_index                 │
  │    ├─ IndexExistsEvent  → load_from_disk   │
  │    └─ BuildIndexEvent   → load_documents   │
  │                          → chunk_nodes     │
  │                          → embed_and_store │
  └───────────────────────────────────────────┘

  ┌───────────────────────────────────────────┐
  │  RouterWorkflow  (per query)               │
  │  StartEvent → classify                    │
  │    ├─ SemanticQueryEvent                  │
  │    │     → RAGQueryWorkflow               │
  │    │           → retrieve → postprocess  │
  │    │           → synthesize / handle_empty│
  │    └─ StructuredQueryEvent               │
  │          → structured_query              │
  │          → Cohere synthesize             │
  └───────────────────────────────────────────┘
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import gradio as gr

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lazy globals (initialised once at startup) ────────────────────────────────
_workflow = None  # RouterWorkflow | None  – lives for the lifetime of the process


def _init_workflow(reindex: bool, project_roots: list[str]) -> None:
    """
    Bootstrap step:
      1. Run IndexingWorkflow (builds or loads FAISS index).
      2. Run extraction (builds or loads structured_store/items.json).
      3. Create a long-lived RouterWorkflow that is reused for every query.
    """
    global _workflow
    from index_workflow import run_indexing
    from rag_workflow import RAGQueryWorkflow
    from router_workflow import RouterWorkflow
    from extractor import extract_from_documents
    from structured_store import StructuredStore
    from loader import load_documents

    if reindex or project_roots:
        if not project_roots:
            logger.warning(
                "No project roots specified for reindexing. "
                "Edit PROJECT_ROOTS in app.py or pass --roots."
            )
        logger.info("Running IndexingWorkflow (build branch) for: %s", project_roots)
    else:
        logger.info("Running IndexingWorkflow (load-from-disk branch) …")

    index = asyncio.run(
        run_indexing(reindex=reindex, project_roots=project_roots)
    )

    # ── Structured extraction ──────────────────────────────────────────────
    # Re-load raw documents for extraction (cheap: just file reads, no embedding).
    # extract_from_documents() uses a cached JSON file unless force=True.
    # force only when --index was explicitly passed, not just --roots.
    if project_roots:
        documents = load_documents(project_roots)
        items = extract_from_documents(documents, force=reindex)
    else:
        # Load branch: use cached store; if it doesn't exist yet, skip gracefully.
        try:
            items = extract_from_documents([], force=False)
        except Exception as exc:
            logger.warning("[init] Could not load structured store: %s", exc)
            items = []

    store = StructuredStore(items)
    rag = RAGQueryWorkflow(index=index, timeout=60)
    _workflow = RouterWorkflow(rag_workflow=rag, store=store, timeout=90)
    logger.info("RouterWorkflow ready (semantic + structured retrieval active).")


def _reload_engine(new_index) -> None:
    """Hot-swap the RouterWorkflow after a watcher-triggered index rebuild."""
    global _workflow
    from rag_workflow import RAGQueryWorkflow
    from router_workflow import RouterWorkflow
    # Reuse the existing structured store (watcher only rebuilds the vector index)
    existing_store = _workflow._store if _workflow is not None else None
    if existing_store is None:
        from structured_store import StructuredStore
        existing_store = StructuredStore([])
    rag = RAGQueryWorkflow(index=new_index, timeout=60)
    _workflow = RouterWorkflow(rag_workflow=rag, store=existing_store, timeout=90)
    logger.info("RouterWorkflow hot-swapped after index rebuild.")


# ── Gradio callback ───────────────────────────────────────────────────────────

async def answer_question(question: str, history: list) -> tuple[str, list]:
    """Process a user question through RAGQueryWorkflow and return the answer."""
    if not question.strip():
        return "", history

    history = list(history)  # avoid mutating the caller's list

    if _workflow is None:
        history.append({"role": "user", "content": question})
        history.append({
            "role": "assistant",
            "content": "⚠️ System is still initialising. Please wait a moment and try again.",
        })
        return "", history

    from query_engine import QueryResult

    result: QueryResult = await _workflow.run(question=question)

    if result.warning_only:
        warning_md = (
            "⚠️ **Warning**\n\n"
            f"{result.answer}"
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": warning_md})
        return "", history

    sources_md = result.format_sources()
    confidence_note = (
        "> ⚠️ **Low confidence** – the retrieved content may not closely match your "
        "question. Consider rephrasing or asking about a more specific topic.\n\n"
        if result.low_confidence
        else ""
    )
    full_answer = f"{confidence_note}{result.answer}\n\n---\n**📂 Sources:**\n{sources_md}"

    # Gradio 6 messages format: list of {role, content} dicts
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_answer})
    return "", history


def get_index_stats() -> str:
    """Return a short status string shown in the UI."""
    try:
        from pipeline import FAISS_STORE_DIR
        if not FAISS_STORE_DIR.exists():
            return "⚠️ FAISS index not built yet. Run: `python index.py ./sample_project`"
        files = list(FAISS_STORE_DIR.iterdir())
        size_kb = sum(f.stat().st_size for f in files if f.is_file()) // 1024
        return f"✅ FAISS index ready at `faiss_store/` — {size_kb} KB on disk"
    except Exception as exc:
        return f"⚠️ Could not fetch index stats: {exc}"


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLES = [
    "What is the main color chosen for the system design?",
    "Are there any recurring technical constraints across documents?",
    "What database schema changes were made recently?",
    "Which component was flagged as problematic or sensitive?",
    "Is there a consistent guideline about RTL support in the UI?",
    "What languages were decided for the interface translations?",
]


# ── Build Gradio interface ────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic Docs RAG") as demo:

        gr.Markdown(
            """
            # 🔍 Agentic Docs RAG
            Ask questions about the documentation generated by your Agentic Coding tools
            (Cursor, Claude Code, Kiro, …).

            The system searches across **all** `.md` files, finds the most relevant chunks,
            and synthesises an answer with citations.
            """
        )

        stats_box = gr.Markdown(value=get_index_stats)

        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
        )

        question_box = gr.Textbox(
            placeholder="Ask anything about your project's AI documentation…",
            label="Your question",
            scale=8,
            autofocus=True,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

        gr.Examples(
            examples=EXAMPLES,
            inputs=question_box,
            label="Example questions",
        )

        clear_btn = gr.Button("🗑 Clear conversation", variant="secondary")

        # ── Event wiring ─────────────────────────────────────────────────────
        send_btn.click(
            fn=answer_question,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        question_box.submit(
            fn=answer_question,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot],
        )
        clear_btn.click(lambda: [], outputs=[chatbot])  # empty list resets messages-format chatbot

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

# ✏️  Edit this list to point at your own project directories.
PROJECT_ROOTS: list[str] = [
    # "C:/path/to/my-project",
    # "C:/path/to/another-project",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic Docs RAG — Gradio UI")
    parser.add_argument(
        "--index",
        action="store_true",
        help="Re-index documents before launching.",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[],
        metavar="PATH",
        help="Project root directories to index (overrides PROJECT_ROOTS in app.py).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on (default: 7860).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch project roots for .md changes and auto-rebuild the index.",
    )
    args = parser.parse_args()

    roots = args.roots or PROJECT_ROOTS
    _init_workflow(reindex=args.index, project_roots=roots)

    if args.watch:
        from watcher import start_watcher
        start_watcher(project_roots=roots or ["." ], on_rebuilt=_reload_engine)
        logger.info("File watcher active – index will auto-rebuild on .md changes.")

    demo = build_ui()
    try:
        demo.launch(
            server_port=args.port,
            share=False,
            theme=gr.themes.Soft(primary_hue="blue"),
        )
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        logger.warning(
            "Localhost is not accessible in this environment; retrying with share=True."
        )
        demo.launch(
            server_port=args.port,
            share=True,
            theme=gr.themes.Soft(primary_hue="blue"),
        )


if __name__ == "__main__":
    main()
