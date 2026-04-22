"""
extraction_schema.py – Pydantic models for structured items extracted from .md docs.

Five item types are recognised:
  decision   – a technical or architectural decision
  rule       – a coding rule, guideline, or convention
  warning    – a sensitive component or "do not touch" note
  dependency – a specific library, service, or external dependency
  change     – a notable change, update, or migration
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Canonical set of item types
ItemType = Literal["decision", "rule", "warning", "dependency", "change"]


class ExtractedItem(BaseModel):
    """A single structured data point extracted from a Markdown document."""

    type: ItemType
    description: str = Field(description="Clear one-sentence summary of the item")
    source_file: str = Field(description="Filename where this item was found")
    tool: str = Field(description="Agentic coding tool (cursor, claude_code, kiro, …)")
    title: str = Field(description="Section heading that contained this item")
    date_extracted: str = Field(description="ISO 8601 timestamp of extraction")
    raw_text: str = Field(description="Original text snippet (max 200 chars)")
