# Agentic Docs RAG

A **Retrieval-Augmented Generation** system that indexes the Markdown documentation produced by Agentic Coding tools (Cursor, Claude Code, Kiro, Windsurf, Cline, …) and lets you ask natural-language questions about them.

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
   PineconeVectorStore       (pipeline.py)
        │  stores vectors + metadata
        ▼
   VectorStoreIndex
        │
        ▼
   RetrieverQueryEngine      (query_engine.py)
        │  top-k cosine similarity retrieval
        ▼
   CohereLLM (command-r-plus)
        │  synthesises answer from retrieved context
        ▼
   Gradio UI                 (app.py)
```

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A [Cohere](https://cohere.com) account (free tier works)
- A [Pinecone](https://pinecone.io) account (free Serverless index works)

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure credentials
```bash
copy .env.example .env
# then edit .env with your API keys
```

### 4. Index your documents
```bash
# Point at your real project(s) — or use the bundled sample data
python index.py sample_project

# Or index multiple projects at once
python index.py C:\path\to\project1 C:\path\to\project2
```

### 5. Launch the UI
```bash
python app.py
# Open http://localhost:7860
```

Or combine both steps:
```bash
python app.py --index --roots sample_project
```

---

## Project Structure

```
rag/
├── .env.example          # Template — copy to .env
├── requirements.txt      # Python dependencies
│
├── config.py             # Centralised settings (reads .env)
├── loader.py             # Discovers & loads .md files with metadata
├── pipeline.py           # Chunk → embed → Pinecone indexing pipeline
├── query_engine.py       # LlamaIndex query engine + result formatting
├── app.py                # Gradio web UI
├── index.py              # CLI entry-point for (re-)indexing
│
└── sample_project/       # Demo data mimicking real tool output
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
