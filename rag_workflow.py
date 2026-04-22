"""
rag_workflow.py – LlamaIndex event-driven workflow for per-query RAG.

Event flow
----------
    StartEvent(question)
        │
        ▼
    [retrieve]  ──► RetrievedEvent
                         │
                         ▼
                    [postprocess]
                         │
                         ├─ nodes remain ──► PostprocessedEvent
                         │                        │
                         │                        ▼
                         │                   [synthesize] ──► StopEvent(QueryResult)
                         │
                         └─ all filtered   ──► EmptyResultEvent
                                                  │
                                                  ▼
                                           [handle_empty] ──► StopEvent(QueryResult)

Usage
-----
    from rag_workflow import run_query, RAGQueryWorkflow
    from index_workflow import run_indexing
    import asyncio

    index   = asyncio.run(run_indexing())
    result  = asyncio.run(run_query(index, "What is the main colour?"))
    print(result.answer)

    # Or keep the workflow instance alive to avoid re-creating the LLM each call:
    workflow = RAGQueryWorkflow(index=index, timeout=60)
    result   = asyncio.run(workflow.run(question="What is the main colour?"))
"""
from __future__ import annotations

import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.postprocessor import LongContextReorder, SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.llms.cohere import Cohere

import config
from query_engine import SYSTEM_PROMPT, QueryResult, ValidationError, validate_question

logger = logging.getLogger(__name__)


# ── Events ─────────────────────────────────────────────────────────────────────

class RetrievedEvent(Event):
    """
    FAISS vector search completed.

    Payload
    -------
    question : str              – the original user question
    nodes    : list[NodeWithScore] – raw top-K candidates from FAISS
    """
    question: str
    nodes: list   # list[NodeWithScore]


class PostprocessedEvent(Event):
    """
    Nodes survived similarity cutoff + LongContextReorder.

    Payload
    -------
    question       : str                – unchanged question
    nodes          : list[NodeWithScore] – filtered & reordered nodes
    low_confidence : bool               – True when best score < LOW_CONFIDENCE_THRESHOLD
    """
    question: str
    nodes: list         # list[NodeWithScore]
    low_confidence: bool


class EmptyResultEvent(Event):
    """
    Every retrieved node was below the similarity cutoff – nothing relevant.

    Payload
    -------
    question : str – the question that yielded no results (for logging)
    """
    question: str


class InvalidQuestionEvent(Event):
    """
    The question failed input validation – skip retrieval entirely.

    Payload
    -------
    question : str – the raw (invalid) question string
    reason   : str – user-facing explanation of why it was rejected
    """
    question: str
    reason: str


# ── Workflow ───────────────────────────────────────────────────────────────────

class RAGQueryWorkflow(Workflow):
    """
    Event-driven retrieve → postprocess → synthesize pipeline.

    The postprocess step branches the event graph:
      • nodes survive  → PostprocessedEvent → synthesize
      • all filtered   → EmptyResultEvent   → handle_empty

    Parameters
    ----------
    index   : VectorStoreIndex – a loaded FAISS index (from index_workflow).
    timeout : float            – seconds before the workflow raises TimeoutError.
    """

    def __init__(self, index: VectorStoreIndex, **kwargs):
        super().__init__(**kwargs)
        self._retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=config.TOP_K,
        )
        self._postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=config.SIMILARITY_CUTOFF),
            LongContextReorder(),
        ]
        self._synthesizer = get_response_synthesizer(
            llm=Cohere(
                model=config.COHERE_LLM_MODEL,
                api_key=config.COHERE_API_KEY,
                system_prompt=SYSTEM_PROMPT,
            ),
            response_mode="compact",
            verbose=False,
        )

    # ── Step 1: Retrieve ───────────────────────────────────────────────────────

    @step
    async def retrieve(self, ev: StartEvent) -> RetrievedEvent | InvalidQuestionEvent:
        """
        Validate the question, then run a vector search against the FAISS index.

        Input : StartEvent              { question: str }
        Output: RetrievedEvent          { question, nodes } – on valid input
                InvalidQuestionEvent    { question, reason } – on invalid input
        """
        raw_question: str = ev.get("question", "")
        try:
            question = validate_question(raw_question)
        except ValidationError as exc:
            logger.info("[retrieve] Question rejected by validation: %s", exc)
            return InvalidQuestionEvent(question=raw_question, reason=str(exc))

        logger.info("[retrieve] Query: %s", question)
        nodes: list[NodeWithScore] = self._retriever.retrieve(question)
        logger.info("[retrieve] Got %d node(s) from FAISS.", len(nodes))
        return RetrievedEvent(question=question, nodes=nodes)

    # ── Step 2: Postprocess (branching) ────────────────────────────────────────

    @step
    async def postprocess(
        self, ev: RetrievedEvent
    ) -> PostprocessedEvent | EmptyResultEvent:
        """
        Apply SimilarityPostprocessor + LongContextReorder, then branch.

        Input : RetrievedEvent      { question, nodes }
        Output: PostprocessedEvent  { question, nodes }  – if any nodes survive
                EmptyResultEvent    { question }          – if all nodes are pruned
        """
        query_bundle = QueryBundle(query_str=ev.question)
        nodes: list[NodeWithScore] = list(ev.nodes)

        for processor in self._postprocessors:
            nodes = processor.postprocess_nodes(nodes, query_bundle=query_bundle)

        logger.info("[postprocess] %d node(s) remain after filtering.", len(nodes))

        if not nodes:
            logger.info(
                "[postprocess] All nodes pruned → EmptyResultEvent for %r.",
                ev.question,
            )
            return EmptyResultEvent(question=ev.question)

        max_score: float = max((n.score or 0.0 for n in nodes), default=0.0)
        low_confidence = max_score < config.LOW_CONFIDENCE_THRESHOLD
        if low_confidence:
            logger.info(
                "[postprocess] Best score %.3f < LOW_CONFIDENCE_THRESHOLD %.3f "
                "– flagging answer as low-confidence.",
                max_score,
                config.LOW_CONFIDENCE_THRESHOLD,
            )

        return PostprocessedEvent(
            question=ev.question, nodes=nodes, low_confidence=low_confidence
        )

    # ── Step 3a: Synthesize ────────────────────────────────────────────────────

    @step
    async def synthesize(self, ev: PostprocessedEvent) -> StopEvent:
        """
        Synthesize a final answer via the Cohere LLM.

        If confidence is low, or synthesis fails, return a warning-only response
        (no answer content and no source list).

        Input : PostprocessedEvent  { question, nodes }
        Output: StopEvent           { result: QueryResult }
        """
        try:
            response = self._synthesizer.synthesize(ev.question, nodes=ev.nodes)
            answer = str(response).strip()
            logger.info("[synthesize] Synthesis succeeded.")
        except Exception as exc:
            logger.warning("[synthesize] LLM call failed (%s) – warning-only response.", exc)
            return StopEvent(
                result=QueryResult(
                    answer=(
                        "Warning: I couldn't generate a reliable answer right now. "
                        "Please try again in a moment."
                    ),
                    sources=[],
                    warning_only=True,
                )
            )

        if ev.low_confidence:
            answer = (
                "⚠️ _Low confidence – the retrieved content may not fully match your question._\n\n"
                + answer
            )

        return StopEvent(
            result=QueryResult(
                answer=answer,
                sources=_extract_sources(ev.nodes),
                low_confidence=ev.low_confidence,
            )
        )

    # ── Step 3b: Handle empty result ───────────────────────────────────────────

    @step
    async def handle_empty(self, ev: EmptyResultEvent) -> StopEvent:
        """
        Return a graceful 'not found' answer when no relevant chunks survive.

        Input : EmptyResultEvent  { question }
        Output: StopEvent         { result: QueryResult }
        """
        logger.info("[handle_empty] No relevant content for %r.", ev.question)
        return StopEvent(
            result=QueryResult(
                answer=(
                    "Warning: I couldn't find relevant information in the indexed "
                    "documentation for this question."
                ),
                sources=[],
                warning_only=True,
            )
        )

    # ── Step 3c: Handle invalid question ──────────────────────────────────────

    @step
    async def handle_invalid_question(self, ev: InvalidQuestionEvent) -> StopEvent:
        """
        Return a validation error message without touching the FAISS index.

        Input : InvalidQuestionEvent  { question, reason }
        Output: StopEvent             { result: QueryResult }
        """
        logger.info(
            "[handle_invalid_question] Rejected %r – %s", ev.question, ev.reason
        )
        return StopEvent(
            result=QueryResult(
                answer=ev.reason,
                sources=[],
                warning_only=True,
            )
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_sources(nodes: list[NodeWithScore]) -> list[dict]:
    """Convert a list of NodeWithScore objects into serialisable source dicts."""
    sources: list[dict] = []
    for nws in nodes:
        meta = nws.node.metadata or {}
        sources.append(
            {
                "tool":          meta.get("tool", "unknown"),
                "file_name":     meta.get("file_name", "?"),
                "file_path":     meta.get("file_path", "?"),
                "title":         meta.get("title", ""),
                "relative_path": meta.get("relative_path", "?"),
                "score":         nws.score,
                "text_snippet":  nws.node.get_content()[:300],
            }
        )
    return sources


# ── Public helper ───────────────────────────────────────────────────────────────

async def run_query(index: VectorStoreIndex, question: str) -> QueryResult:
    """
    Convenience async entry point: instantiate the workflow and run one query.

    For repeated queries prefer keeping a ``RAGQueryWorkflow`` instance alive
    (avoids reconstructing the Cohere LLM client on every call).
    """
    workflow = RAGQueryWorkflow(index=index, timeout=60)
    return await workflow.run(question=question)
