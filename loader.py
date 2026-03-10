"""
loader.py – Discovers and loads Markdown files produced by Agentic Coding tools.

The loader:
  1. Accepts a list of project root directories to scan.
  2. Identifies which Agentic Coding tool each file belongs to (via folder heuristics).
  3. Returns LlamaIndex Document objects enriched with tool/source metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _detect_tool(file_path: Path, project_root: Path) -> str:
    """Return the name of the Agentic Coding tool that owns *file_path*.

    Detection is based on whether any known tool folder pattern appears in
    the file's path relative to the project root.
    """
    relative = file_path.relative_to(project_root)
    parts = {p.lower() for p in relative.parts}

    for tool, patterns in config.TOOL_FOLDER_MAP.items():
        for pattern in patterns:
            # pattern may be a simple name like ".cursor" or a nested path
            pattern_parts = {p.lower() for p in Path(pattern).parts}
            if pattern_parts & parts:
                return tool

    return "unknown"


def _iter_md_files(directory: Path) -> Iterator[Path]:
    """Recursively yield all .md / .mdx files under *directory*."""
    for ext in ("*.md", "*.mdx"):
        yield from directory.rglob(ext)


# ─────────────────────────────────────────────────────────────────────────────
# Reader
# ─────────────────────────────────────────────────────────────────────────────

class AgenticDocsReader(BaseReader):
    """Reads Markdown files from one or more project roots.

    Parameters
    ----------
    project_roots:
        List of paths to project root directories.
        Each directory is scanned recursively for .md / .mdx files.
    tool_hint:
        Optional explicit tool name to assign to all files loaded through
        this reader instance (overrides auto-detection).
    """

    def __init__(
        self,
        project_roots: list[str | Path],
        tool_hint: str | None = None,
    ) -> None:
        self._roots = [Path(r) for r in project_roots]
        self._tool_hint = tool_hint

    # ------------------------------------------------------------------
    # BaseReader interface
    # ------------------------------------------------------------------

    def load_data(self) -> list[Document]:  # type: ignore[override]
        documents: list[Document] = []

        for root in self._roots:
            if not root.exists():
                logger.warning("Project root does not exist, skipping: %s", root)
                continue

            for md_file in _iter_md_files(root):
                try:
                    text = md_file.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Could not read %s: %s", md_file, exc)
                    continue

                if not text.strip():
                    continue  # skip empty files

                tool = self._tool_hint or _detect_tool(md_file, root)

                metadata = {
                    # ── provenance ──────────────────────────────────────────
                    "tool": tool,
                    "file_name": md_file.name,
                    "file_path": str(md_file),
                    "relative_path": str(md_file.relative_to(root)),
                    "project_root": str(root),
                    # ── content hints ────────────────────────────────────────
                    "title": _extract_title(text, md_file.stem),
                    "char_count": len(text),
                }

                doc = Document(text=text, metadata=metadata)
                documents.append(doc)
                logger.debug("Loaded [%s] %s", tool, md_file)

        logger.info("Loaded %d documents from %d project root(s).", len(documents), len(self._roots))
        return documents


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _extract_title(text: str, fallback: str) -> str:
    """Return the first H1 heading found in *text*, or *fallback*."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_documents(project_roots: list[str | Path]) -> list[Document]:
    """Convenience wrapper around AgenticDocsReader."""
    reader = AgenticDocsReader(project_roots=project_roots)
    return reader.load_data()
