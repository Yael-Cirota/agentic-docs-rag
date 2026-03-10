"""
query_engine.py – Builds a LlamaIndex query engine on top of the Pinecone index.

The engine:
  • Uses Cohere as the LLM for answer synthesis.
  • Retrieves top-k semantically similar chunks.
  • Injects metadata (tool, file, title) into the response so users know
    *which* tool/file each piece of information came from.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.cohere import Cohere

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Custom RAG prompt (Hebrew + English aware)
# ─────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = PromptTemplate(
    """\
You are a helpful assistant that answers questions about an Agentic Coding project \
by analysing documentation written by AI coding tools (Cursor, Claude Code, Kiro, etc.).

Answer the question using ONLY the context provided below.
If the answer is not found in the context, say so clearly.
Always cite which tool and file the information comes from.
You may answer in the same language the question was asked in (Hebrew or English).

Context:
---------------------
{context_str}
---------------------

Question: {query_str}

Answer:"""
)


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
# Engine builder
# ─────────────────────────────────────────────────────────────────────────────

def build_query_engine(index: VectorStoreIndex, top_k: int = 5) -> RetrieverQueryEngine:
    """Return a configured RetrieverQueryEngine backed by the given index."""
    llm = Cohere(
        api_key=config.COHERE_API_KEY,
        model=config.COHERE_LLM_MODEL,
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    response_synthesizer = get_response_synthesizer(
        llm=llm,
        text_qa_template=RAG_SYSTEM_PROMPT,
        response_mode="compact",
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    return engine


# ─────────────────────────────────────────────────────────────────────────────
# High-level query function
# ─────────────────────────────────────────────────────────────────────────────

def query(engine: RetrieverQueryEngine, question: str) -> QueryResult:
    """Run *question* through the engine and return a structured QueryResult."""
    logger.info("Query: %s", question)
    response = engine.query(question)

    sources: list[dict] = []
    for node_with_score in response.source_nodes:
        meta = node_with_score.node.metadata or {}
        sources.append(
            {
                "tool": meta.get("tool", "unknown"),
                "file_name": meta.get("file_name", "?"),
                "file_path": meta.get("file_path", "?"),
                "title": meta.get("title", ""),
                "relative_path": meta.get("relative_path", "?"),
                "score": node_with_score.score,
                "text_snippet": node_with_score.node.get_content()[:300],
            }
        )

    return QueryResult(answer=str(response), sources=sources)
