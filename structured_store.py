"""
structured_store.py – In-memory query layer over the extracted items JSON.

Usage
-----
    from structured_store import StructuredStore
    from extractor import extract_from_documents

    items = extract_from_documents(documents)
    store = StructuredStore(items)

    results = store.query(types=["warning"], keywords=["payment"])
    print(store.format_for_llm(results))
"""
from __future__ import annotations

import logging
from typing import get_args

from extraction_schema import ExtractedItem, ItemType

logger = logging.getLogger(__name__)

_ALL_TYPES: frozenset[str] = frozenset(get_args(ItemType))


class StructuredStore:
    """Thin, queryable wrapper around a list of :class:`ExtractedItem` objects."""

    def __init__(self, items: list[ExtractedItem]) -> None:
        self._items = items
        logger.info("[StructuredStore] Initialised with %d item(s).", len(items))

    @property
    def items(self) -> list[ExtractedItem]:
        return self._items

    # ── Query ────────────────────────────────────────────────────────────────

    def query(
        self,
        *,
        types: list[str] | None = None,
        keywords: list[str] | None = None,
        tool: str | None = None,
        limit: int = 20,
    ) -> list[ExtractedItem]:
        """Filter items by type and/or keywords.

        Parameters
        ----------
        types:
            Restrict to these item types (e.g. ``["warning", "decision"]``).
            Unknown type strings are silently ignored.
        keywords:
            Keep only items whose description, title, or raw_text contains at
            least one of these strings (case-insensitive).
        tool:
            Restrict to a specific agentic coding tool.
        limit:
            Maximum number of results returned (default 20).
        """
        results = list(self._items)

        if types:
            valid = set(types) & _ALL_TYPES
            if valid:
                results = [r for r in results if r.type in valid]

        if tool:
            results = [r for r in results if r.tool == tool.lower()]

        if keywords:
            kw_lower = [k.lower() for k in keywords]
            results = [
                r for r in results
                if any(
                    kw in r.description.lower()
                    or kw in r.raw_text.lower()
                    or kw in r.title.lower()
                    for kw in kw_lower
                )
            ]

        return results[:limit]

    # ── Formatting ───────────────────────────────────────────────────────────

    def format_for_llm(self, items: list[ExtractedItem]) -> str:
        """Format items as a readable context block for the synthesizer LLM."""
        if not items:
            return "No matching items found in the structured store."

        lines: list[str] = []
        for i, item in enumerate(items, 1):
            lines.append(
                f"{i}. [{item.type.upper()}] {item.description}\n"
                f"   Source: {item.source_file}  (tool: {item.tool})\n"
                f"   Section: {item.title}\n"
                f"   Excerpt: {item.raw_text}"
            )
        return "\n\n".join(lines)
