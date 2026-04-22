# Agentic Docs RAG

A **Retrieval-Augmented Generation** system that indexes the Markdown documentation produced by Agentic Coding tools (Cursor, Claude Code, Kiro, Windsurf, Cline, …) and lets you ask natural-language questions about them.

Queries are handled by an event-driven **RouterWorkflow** that classifies each question and routes it to either a semantic vector search (FAISS + Cohere embeddings) or a structured item store (pre-extracted decisions, rules, warnings, and dependencies).

---

## Architecture

```
project repo
├── .cursor/rules/*.md      ← Cursor rules
├── .claude/CLAUDE.md       ← Claude Code instructions
├── .kiro/steering/*.md     ← Kiro steering docs
└── .kiro/specs/*.md        ← Kiro feature specs
        │
        ▼
   AgenticDocsReader         (loader.py)
        │  loads + adds metadata: tool, file, title
        ▼
   MarkdownNodeParser        (pipeline.py)
        │  splits on headings
        ▼
   SentenceSplitter          (pipeline.py)
        │  caps chunks at CHUNK_SIZE tokens
        ▼
   CohereEmbedding           embed-multilingual-v3.0
        │  1024-dim vectors, Hebrew+English aware
        ▼
   FaissVectorStore          (index_workflow.py)        ← persisted to faiss_store/
        │  stores vectors + metadata locally
        ▼
   VectorStoreIndex
        │
        ▼
   RouterWorkflow            (router_workflow.py)
        │
        ├─ semantic query ──► RAGQueryWorkflow          (rag_workflow.py)
        │                       retrieve → postprocess → synthesize
        │                       CohereLLM (command-r-plus)
        │
        └─ structured query ► StructuredStore           (structured_store.py)
                                filter by type / keyword
                                CohereLLM synthesizes answer
        ▼
   Gradio UI                 (app.py)
```

### Indexing Workflow (`index_workflow.py`)

```
StartEvent(reindex, project_roots)
    │
    ▼
[check_index]
    ├─ store exists & reindex=False ──► [load_from_disk] ──► StopEvent(index)
    └─ store missing or reindex=True ──► [load_documents]
                                              │
                                         [chunk_nodes]
                                              │
                                         [embed_and_store] ──► StopEvent(index)
```

### RAG Query Workflow (`rag_workflow.py`)

```
StartEvent(question)
    ▼
[retrieve] ──► [postprocess]
                    ├─ nodes remain ──► [synthesize] ──► StopEvent(QueryResult)
                    └─ all filtered  ──► [handle_empty] ──► StopEvent(QueryResult)
```

### Structured Extraction (`extractor.py` + `extraction_schema.py`)

The Cohere LLM scans each loaded document and extracts typed items:

| Type | Description |
|---|---|
| `decision` | Technical or architectural decision |
| `rule` | Coding rule, guideline, or convention |
| `warning` | Sensitive component or "do not touch" note |
| `dependency` | Library, service, or external dependency |
| `change` | Notable change, update, or migration |

Extracted items are cached in `structured_store/items.json` and queried by `StructuredStore`.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A [Cohere](https://cohere.com) account (free tier works) — required
- [Pinecone](https://pinecone.io) — **optional** (FAISS is used by default, no cloud account needed)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
```bash
copy .env.example .env
# then edit .env with your API keys
```

Only `COHERE_API_KEY` is required. All other settings have sensible defaults.

### 4. Index your documents
```bash
# Point at your real project(s) — or use the bundled sample data
python index.py sample_project

# Or index multiple projects at once
python index.py C:\path\to\project1 C:\path\to\project2
```

The FAISS index is persisted to `faiss_store/` and reused on subsequent runs.

### 5. Launch the UI
```bash
python app.py
# Open http://localhost:7860
```

Or combine indexing + UI in one step:
```bash
python app.py --index --roots sample_project
```

### 6. (Optional) Live file watching
The watcher rebuilds the index automatically whenever a `.md` or `.mdx` file changes:
```bash
python watcher.py sample_project
# or watch multiple roots:
python watcher.py C:\path\to\project1 C:\path\to\project2
```

When used via `app.py`, the watcher hot-swaps the live query engine without a server restart.

---

## Configuration

All settings are read from `.env` (or environment variables). Defaults are shown below.

| Variable | Default | Description |
|---|---|---|
| `COHERE_API_KEY` | — | **Required.** Cohere API key |
| `PINECONE_API_KEY` | — | Optional. Only needed if using Pinecone |
| `COHERE_EMBED_MODEL` | `embed-multilingual-v3.0` | Embedding model |
| `COHERE_LLM_MODEL` | `command-r-plus-08-2024` | LLM for synthesis |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `SIMILARITY_CUTOFF` | `0.35` | Minimum similarity score to keep a chunk |

---

## Project Structure

```
rag/
├── .env.example              # Template — copy to .env
├── requirements.txt          # Python dependencies
│
├── config.py                 # Centralised settings (reads .env)
├── loader.py                 # Discovers & loads .md files with metadata
├── pipeline.py               # Chunk → embed helpers used by index_workflow
├── index_workflow.py         # LlamaIndex Workflow: build or load FAISS index
├── rag_workflow.py           # LlamaIndex Workflow: per-query RAG pipeline
├── router_workflow.py        # LlamaIndex Workflow: classify & route each query
├── query_engine.py           # QueryResult model, validation, system prompt
├── extraction_schema.py      # Pydantic models for structured extracted items
├── extractor.py              # Cohere-powered structured extraction from docs
├── structured_store.py       # In-memory filter/query layer over extracted items
├── watcher.py                # watchdog-based live file watcher & index rebuilder
├── app.py                    # Gradio web UI
├── index.py                  # CLI entry-point for (re-)indexing
│
├── faiss_store/              # Persisted FAISS index (auto-created)
├── structured_store/         # Cached extracted items JSON (auto-created)
└── sample_project/           # Demo data mimicking real agentic tool output
```
    ├── .cursor/rules/
    ├── .claude/
    └── .kiro/
```

---

## Supported Tools & Their Folders

| Tool | Default folder(s) |
|---|---|
| Cursor | `.cursor/`, `.cursorrules` |
| Claude Code | `.claude/`, `CLAUDE.md` |
| Kiro | `.kiro/specs/`, `.kiro/steering/`, `.kiro/hooks/` |
| GitHub Copilot | `.github/copilot-instructions.md` |
| Cline | `.clinerules/`, `.cline/` |
| Windsurf | `.windsurf/`, `.windsurfrules` |

To add a new tool, edit `TOOL_FOLDER_MAP` in `config.py`.

---

## Example Questions

- *"What is the primary color chosen for the UI?"*
- *"Which database schema changes were made recently?"*
- *"Are there any recurring technical constraints across documents?"*
- *"Which component was flagged as sensitive or problematic?"*
- *"Is there a consistent RTL guideline?"*
- *"What languages are supported in the interface?"*

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `COHERE_API_KEY` | — | **Required** |
| `PINECONE_API_KEY` | — | **Required** |
| `PINECONE_INDEX_NAME` | `agentic-docs` | Pinecone index name |
| `PINECONE_CLOUD` | `aws` | Cloud provider |
| `PINECONE_REGION` | `us-east-1` | Region |
| `COHERE_EMBED_MODEL` | `embed-multilingual-v3.0` | Handles Hebrew + English |
| `COHERE_LLM_MODEL` | `command-r-plus` | For answer synthesis |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
