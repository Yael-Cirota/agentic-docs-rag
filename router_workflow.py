"""
router_workflow.py – Top-level LlamaIndex Workflow that classifies each query
and routes it to either semantic (RAGQueryWorkflow) or structured retrieval.

Event flow
----------
    StartEvent(question)
        │
        ▼
    [classify]
        ├─ SemanticQueryEvent   → [semantic_query]   → StopEvent(QueryResult)
        └─ StructuredQueryEvent → [structured_query] → StopEvent(QueryResult)

Routing heuristic
-----------------
    "Give me all …", "List all …", "What warnings exist?" → structured
    "How does X work?", "What is the guideline for Y?"    → semantic

Usage
-----
    from router_workflow import RouterWorkflow
    from rag_workflow import RAGQueryWorkflow
    from structured_store import StructuredStore

    rag      = RAGQueryWorkflow(index=index, timeout=60)
    store    = StructuredStore(items)
    router   = RouterWorkflow(rag_workflow=rag, store=store, timeout=90)
    result   = await router.run(question="List all technical decisions")
"""
from __future__ import annotations

import json
import logging

import cohere
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

import config
from query_engine import QueryResult, ValidationError, validate_question
from rag_workflow import RAGQueryWorkflow
from structured_store import StructuredStore

logger = logging.getLogger(__name__)


# ── Events ─────────────────────────────────────────────────────────────────────

class SemanticQueryEvent(Event):
    """Route to the existing RAGQueryWorkflow (vector similarity search)."""
    question: str


class StructuredQueryEvent(Event):
    """Route to the StructuredStore (pre-extracted decisions, rules, warnings, …)."""
    question: str
    suggested_types: list    # list[str] – subset of ItemType literals
    suggested_keywords: list  # list[str]


# ── Classifier prompt ──────────────────────────────────────────────────────────

_ROUTER_PROMPT = """\
You are a query router for a documentation RAG system.

Decide whether the user's question is best answered via:
  A) "semantic"    – similarity search over raw documentation text
     (use for: "how does X work?", "what is the guideline for Y?", conceptual questions)
  B) "structured"  – lookup in a catalogue of pre-extracted decisions, rules, warnings,
                     dependencies, and changes
     (use for: "list all decisions", "what warnings exist?", "what dependencies are there?",
               "what was marked sensitive?", "what changed?", recency / list questions)

User question: "{question}"

Respond with ONLY a JSON object — no prose, no markdown fences:
{{"route": "semantic" or "structured", "types": [zero or more of: decision, rule, warning, dependency, change], "keywords": [1-3 key terms from the question]}}
"""


# ── Workflow ───────────────────────────────────────────────────────────────────

class RouterWorkflow(Workflow):
    """
    Classifies each user query and delegates to:
      • RAGQueryWorkflow  – for semantic / conceptual questions
      • StructuredStore   – for list / recency / aggregation questions
    """

    def __init__(
        self,
        rag_workflow: RAGQueryWorkflow,
        store: StructuredStore,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._rag = rag_workflow
        self._store = store
        self._cohere = cohere.ClientV2(api_key=config.COHERE_API_KEY)

    # ── Step 1: Classify ──────────────────────────────────────────────────────

    @step
    async def classify(self, ev: StartEvent) -> SemanticQueryEvent | StructuredQueryEvent:
        """
        Use the Cohere LLM to decide which retrieval strategy fits the query.

        Input : StartEvent { question: str }
        Output: SemanticQueryEvent  |  StructuredQueryEvent
        """
        raw_question: str = ev.get("question", "")

        # Validate early; invalid questions fall through to the semantic path
        # which will reject them with a user-friendly message.
        try:
            question = validate_question(raw_question)
        except ValidationError:
            return SemanticQueryEvent(question=raw_question)

        route = "semantic"
        types: list = []
        keywords: list = []

        try:
            resp = self._cohere.chat(
                model=config.COHERE_LLM_MODEL,
                messages=[{"role": "user", "content": _ROUTER_PROMPT.format(question=question)}],
            )
            raw = resp.message.content[0].text.strip()

            # Strip optional ```json … ``` fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            route = str(data.get("route", "semantic")).lower()
            types = list(data.get("types", []))
            keywords = list(data.get("keywords", []))

        except Exception as exc:
            logger.warning(
                "[classify] Classification failed (%s) → defaulting to semantic.", exc
            )

        logger.info(
            "[classify] route=%s  types=%s  keywords=%s", route, types, keywords
        )

        if route == "structured":
            return StructuredQueryEvent(
                question=question,
                suggested_types=types,
                suggested_keywords=keywords,
            )
        return SemanticQueryEvent(question=question)

    # ── Step 2a: Semantic path ────────────────────────────────────────────────

    @step
    async def semantic_query(self, ev: SemanticQueryEvent) -> StopEvent:
        """
        Delegate to the long-lived RAGQueryWorkflow instance.

        Input : SemanticQueryEvent { question }
        Output: StopEvent          { result: QueryResult }
        """
        result: QueryResult = await self._rag.run(question=ev.question)
        return StopEvent(result=result)

    # ── Step 2b: Structured path ──────────────────────────────────────────────

    @step
    async def structured_query(self, ev: StructuredQueryEvent) -> StopEvent:
        """
        Query the StructuredStore then synthesize an answer with Cohere.

        Input : StructuredQueryEvent { question, suggested_types, suggested_keywords }
        Output: StopEvent            { result: QueryResult }
        """
        items = self._store.query(
            types=ev.suggested_types or None,
            keywords=ev.suggested_keywords or None,
        )

        if not items:
            return StopEvent(result=QueryResult(
                answer=(
                    "I couldn't find any matching items in the structured catalogue. "
                    "Try rephrasing or use a more specific keyword."
                ),
                sources=[],
                warning_only=True,
            ))

        context = self._store.format_for_llm(items)
        synthesis_prompt = (
            f"The user asked: {ev.question}\n\n"
            "Here are the relevant structured items extracted from the project "
            "documentation:\n\n"
            f"{context}\n\n"
            "Please write a clear, direct answer based only on the information above."
        )

        try:
            resp = self._cohere.chat(
                model=config.COHERE_LLM_MODEL,
                messages=[{"role": "user", "content": synthesis_prompt}],
            )
            answer = resp.message.content[0].text.strip()
        except Exception as exc:
            logger.warning(
                "[structured_query] LLM synthesis failed (%s) – returning raw items.", exc
            )
            answer = context  # graceful fallback: show extracted items directly

        # Build a deduplicated source list
        seen: set[tuple[str, str]] = set()
        unique_sources: list[dict] = []
        for item in items:
            key = (item.tool, item.source_file)
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "tool": item.tool,
                    "file_name": item.source_file,
                    "title": item.title,
                    "score": None,
                })

        return StopEvent(result=QueryResult(
            answer=f"📋 _{len(items)} structured item(s) matched_\n\n{answer}",
            sources=unique_sources,
        ))
