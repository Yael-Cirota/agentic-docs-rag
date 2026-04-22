"""
extractor.py – Extracts structured items from loaded LlamaIndex Documents using
the Cohere LLM and persists them to structured_store/items.json.

Usage
-----
    from loader import load_documents
    from extractor import extract_from_documents

    documents = load_documents(["./sample_project"])
    items = extract_from_documents(documents, force=True)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import cohere

import config
from extraction_schema import ExtractedItem

logger = logging.getLogger(__name__)

# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are a structured data extraction assistant.

Read the Markdown content below (file: "{filename}", agentic tool: {tool}) and \
extract every notable item that belongs to one of these categories:

  decision   – a technical or architectural decision that was made
  rule       – a coding rule, guideline, or convention to follow
  warning    – a warning, a sensitive component, or a "do not touch" note
  dependency – a specific library, service, or external dependency mentioned
  change     – a notable change, update, or migration described

For EACH item found, emit one JSON object with exactly these keys:
  "type"        – one of: decision / rule / warning / dependency / change
  "description" – a clear one-sentence summary
  "title"       – the nearest Markdown heading above the item (or filename if none)
  "raw_text"    – the verbatim excerpt (≤ 200 characters)

Return ONLY a valid JSON array.  If nothing relevant is found, return [].

--- CONTENT ---
{text}
"""


# ── Public entry point ────────────────────────────────────────────────────────

def extract_from_documents(
    documents: list,
    force: bool = False,
) -> list[ExtractedItem]:
    """Extract structured items from Documents and persist to JSON.

    Parameters
    ----------
    documents:
        List of LlamaIndex ``Document`` objects (with ``text`` and ``metadata``).
    force:
        If True, re-run extraction even if the store file already exists.

    Returns
    -------
    List of :class:`ExtractedItem` instances.
    """
    store_path = Path(config.STRUCTURED_STORE_PATH)

    if not force and store_path.exists():
        logger.info("[extractor] Store already exists – loading from %s", store_path)
        return _load(store_path)

    logger.info("[extractor] Extracting structured items from %d document(s) …", len(documents))
    client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
    now = datetime.now(timezone.utc).isoformat()
    all_items: list[ExtractedItem] = []

    for doc in documents:
        filename = doc.metadata.get("file_name", "unknown")
        tool = doc.metadata.get("tool", "unknown")
        # Cap text to keep well within token budget (~4 000 tokens ≈ 16 000 chars)
        text = doc.text[:4000]

        prompt = _EXTRACTION_PROMPT.format(filename=filename, tool=tool, text=text)

        try:
            resp = client.chat(
                model=config.COHERE_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.message.content[0].text.strip()

            # Strip optional ```json … ``` fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            items_data: list[dict] = json.loads(raw)

            for d in items_data:
                item = ExtractedItem(
                    type=d.get("type", "rule"),
                    description=str(d.get("description", "")).strip(),
                    source_file=filename,
                    tool=tool,
                    title=str(d.get("title", filename)).strip(),
                    date_extracted=now,
                    raw_text=str(d.get("raw_text", ""))[:200],
                )
                all_items.append(item)

        except Exception as exc:
            logger.warning("[extractor] Skipping %s – %s", filename, exc)
            continue

    logger.info("[extractor] Extracted %d item(s) total.", len(all_items))
    _save(store_path, all_items)
    return all_items


# ── Persistence helpers ───────────────────────────────────────────────────────

def _save(path: Path, items: list[ExtractedItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([i.model_dump() for i in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[extractor] Saved %d item(s) to %s", len(items), path)


def _load(path: Path) -> list[ExtractedItem]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [ExtractedItem(**d) for d in data]
    logger.info("[extractor] Loaded %d item(s) from %s", len(items), path)
    return items
