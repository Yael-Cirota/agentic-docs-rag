"""
query_engine.py – Retrieval + Postprocessing + Synthesis via LlamaIndex.

Pipeline:
  1. VectorIndexRetriever        – FAISS top-K search
  2. SimilarityPostprocessor     – drop chunks below score threshold
  3. LongContextReorder          – put best chunks first & last
  4. ResponseSynthesizer         – LlamaIndex compact synthesis via Cohere LLM
  Fallback: raw retrieved chunks when the LLM endpoint is unreachable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import LongContextReorder, SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.llms.cohere import Cohere

import config

logger = logging.getLogger(__name__)


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

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about an Agentic Coding project \
by analysing documentation written by AI coding tools (Cursor, Claude Code, Kiro, etc.).

Answer the question using ONLY the context provided below.
If the answer is not found in the context, say so clearly.
Always cite which tool and file the information comes from.
You may answer in the same language the question was asked in (Hebrew or English).\
"""


@dataclass
class QueryEngine:
    """Bundles retriever + postprocessors + synthesizer."""
    retriever: VectorIndexRetriever
    postprocessors: list
    synthesizer: object  # ResponseSynthesizer


def build_query_engine(index: VectorStoreIndex) -> QueryEngine:
    """Build and return a QueryEngine with postprocessing and LlamaIndex synthesis."""
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=config.TOP_K,
    )

    postprocessors = [
        SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF),
        LongContextReorder(),
    ]

    llm = Cohere(
        model=config.COHERE_LLM_MODEL,
        api_key=config.COHERE_API_KEY,
        system_prompt=SYSTEM_PROMPT,
    )

    synthesizer = get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        verbose=False,
    )

    return QueryEngine(
        retriever=retriever,
        postprocessors=postprocessors,
        synthesizer=synthesizer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: raw chunks display
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


# ─────────────────────────────────────────────────────────────────────────────
# High-level query function
# ─────────────────────────────────────────────────────────────────────────────

def query(engine: QueryEngine, question: str) -> QueryResult:
    """Retrieve → postprocess → synthesize."""
    logger.info("Query: %s", question)

    # 1. Retrieve
    nodes: list[NodeWithScore] = engine.retriever.retrieve(question)
    logger.info("Retrieved %d nodes from FAISS.", len(nodes))

    # 2. Postprocess
    query_bundle = QueryBundle(query_str=question)
    for processor in engine.postprocessors:
        nodes = processor.postprocess_nodes(nodes, query_bundle=query_bundle)
    logger.info("After postprocessing: %d nodes remain.", len(nodes))

    # 3. Synthesize
    try:
        response = engine.synthesizer.synthesize(question, nodes=nodes)
        answer = str(response).strip()
        logger.info("Synthesis succeeded.")
    except Exception as exc:
        logger.warning("Synthesis failed (%s), falling back to raw chunks.", exc)
        answer = _raw_chunk_fallback(nodes)

    # 4. Build sources from postprocessed nodes
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
