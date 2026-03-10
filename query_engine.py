"""
query_engine.py – Retrieval with LlamaIndex FAISS + synthesis with Cohere SDK.

Synthesis: Cohere ClientV2 → POST /v2/chat (command-r-plus-08-2024)
Fallback:  raw retrieved chunks when the LLM endpoint is unreachable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cohere

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore

import config

logger = logging.getLogger(__name__)

# Cohere client
_co_v2: cohere.ClientV2 | None = None


def _get_v2_client() -> cohere.ClientV2:
    global _co_v2
    if _co_v2 is None:
        _co_v2 = cohere.ClientV2(api_key=config.COHERE_API_KEY)
    return _co_v2


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)

    def format_sources(self) -> str:
        if not self.sources:
            return "_No sources found._"
        lines = []
        for i, src in enumerate(self.sources, 1):
            tool = src.get("tool", "unknown")
            file_name = src.get("file_name", "?")
            title = src.get("title", "")
            score = src.get("score", None)
            score_str = f"  (score: {score:.3f})" if score is not None else ""
            lines.append(f"{i}. **[{tool}]** `{file_name}` – {title}{score_str}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Retriever builder
# ─────────────────────────────────────────────────────────────────────────────

def build_query_engine(index: VectorStoreIndex, top_k: int = 5) -> VectorIndexRetriever:
    """Return a VectorIndexRetriever; synthesis is handled separately via Cohere SDK."""
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Synthesis via Cohere SDK v2
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about an Agentic Coding project \
by analysing documentation written by AI coding tools (Cursor, Claude Code, Kiro, etc.).

Answer the question using ONLY the context provided below.
If the answer is not found in the context, say so clearly.
Always cite which tool and file the information comes from.
You may answer in the same language the question was asked in (Hebrew or English).\
"""


def _synthesize(question: str, nodes: list[NodeWithScore]) -> str:
    """Call Cohere /v2/chat with retrieved context.
    Falls back to a formatted chunk display if the LLM endpoint is unreachable.
    """
    context_parts = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        tool = meta.get("tool", "unknown")
        fname = meta.get("file_name", "?")
        context_parts.append(f"[{tool} | {fname}]\n{nws.node.get_content()}")

    context_str = "\n\n---\n\n".join(context_parts) if context_parts else "(no context found)"

    user_message = (
        f"Context:\n---------------------\n{context_str}\n---------------------\n\n"
        f"Question: {question}"
    )

    # ── Attempt 1: /v2/chat ────────────────────────────────────────────────
    try:
        co = _get_v2_client()
        response = co.chat(
            model=config.COHERE_LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        logger.info("Synthesis via /v2/chat succeeded.")
        return response.message.content[0].text
    except Exception as exc_v2:
        logger.warning("/v2/chat unavailable (%s), trying /v1/generate …", exc_v2)


    # ── Fallback: raw retrieved chunks ───────────────────────────────────────
    if not nodes:
        return "No relevant content found for your question."
    lines = ["**⚠️ LLM synthesis unavailable – showing retrieved chunks directly:**\n"]
    for i, nws in enumerate(nodes, 1):
        meta = nws.node.metadata or {}
        tool = meta.get("tool", "unknown")
        fname = meta.get("file_name", "?")
        snippet = nws.node.get_content()[:600].strip()
        lines.append(f"**Chunk {i}** · `{tool}` · `{fname}`\n\n{snippet}\n")
    return "\n---\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# High-level query function
# ─────────────────────────────────────────────────────────────────────────────

def query(retriever: VectorIndexRetriever, question: str) -> QueryResult:
    """Retrieve relevant nodes then synthesise an answer with Cohere v2."""
    logger.info("Query: %s", question)

    nodes: list[NodeWithScore] = retriever.retrieve(question)

    answer = _synthesize(question, nodes)

    sources: list[dict] = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        sources.append(
            {
                "tool": meta.get("tool", "unknown"),
                "file_name": meta.get("file_name", "?"),
                "file_path": meta.get("file_path", "?"),
                "title": meta.get("title", ""),
                "relative_path": meta.get("relative_path", "?"),
                "score": nws.score,
                "text_snippet": nws.node.get_content()[:300],
            }
        )

    return QueryResult(answer=answer, sources=sources)
