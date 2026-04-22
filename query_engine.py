"""
query_engine.py – Shared types and helpers for the RAG query pipeline.

NOTE
----
QueryEngine, build_query_engine() and query() have been moved to
rag_workflow.py as part of the event-driven workflow refactor.
This module now only contains:
  • QueryResult     – dataclass returned by every query
  • SYSTEM_PROMPT   – Cohere system prompt string
  • _raw_chunk_fallback – plain-text fallback when the LLM is unavailable
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    low_confidence: bool = False
    warning_only: bool = False

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
# Input validation
# ─────────────────────────────────────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when a user question fails input validation."""


_WORD_RE = re.compile(r"\w", re.UNICODE)


def validate_question(question: str) -> str:
    """
    Normalise and validate a user question.
    Returns the cleaned question string on success.
    Raises ValidationError with a user-facing message on failure.
    """
    q = question.strip()
    if len(q) < config.MIN_QUESTION_LENGTH:
        raise ValidationError(
            f"Your question is too short – please enter at least "
            f"{config.MIN_QUESTION_LENGTH} characters."
        )
    if not _WORD_RE.search(q):
        raise ValidationError(
            "Your question doesn't contain any recognisable words. "
            "Please rephrase it."
        )
    if len(q) > config.MAX_QUESTION_LENGTH:
        logger.warning(
            "Question truncated from %d to %d characters.",
            len(q),
            config.MAX_QUESTION_LENGTH,
        )
        q = q[: config.MAX_QUESTION_LENGTH]
    return q


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about an Agentic Coding project \
by analysing documentation written by AI coding tools (Cursor, Claude Code, Kiro, etc.).

Answer the question using ONLY the context provided below.
If the answer is not found in the context, say so clearly.
Always cite which tool and file the information comes from.
You may answer in the same language the question was asked in (Hebrew or English).\
"""


# ─────────────────────────────────────────────────────────────────────────────
# Fallback helper
# ─────────────────────────────────────────────────────────────────────────────

def _raw_chunk_fallback(nodes: list[NodeWithScore]) -> str:
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


# (QueryEngine, build_query_engine, query removed – see rag_workflow.py)
