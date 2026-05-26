"""
Agentic RAG — Claude-driven multi-step retrieval loop

Instead of a single retrieve→generate pass, Claude is given tools and
decides WHAT to search for, HOW MANY times, and WHEN it has enough
context to answer. This enables:

  - Multi-hop questions:     search A → find a fact → search B using that fact
  - Comparative questions:   search doc1 → search doc2 → synthesize both
  - Self-correcting:         weak first search → retry with different terms
  - Document-aware routing:  list_documents() → pick relevant source → search
  - Full-section retrieval:  get_section() when top chunks aren't enough

Architecture
────────────
  User question
      ↓
  Claude (with tools) ──tool_use──► execute tool ──tool_result──► Claude
      ↑_______________repeat until stop_reason == "end_turn"_______________↑
      ↓
  Final answer (with citations)

Two entry points:
  answer_agent()  — blocking, returns full dict (used by evals)
  stream_agent()  — generator, yields ndjson lines (used by /query/agent/stream)
"""

import json
import os

import anthropic
from dotenv import load_dotenv
from qdrant_client.models import Filter, FieldCondition, MatchValue

# load_dotenv FIRST — must run before any SDK reads env vars
load_dotenv()

from db import qdrant, COLLECTION_NAME
from ingest import list_documents
from query import retrieve_chunks

# ── Langfuse tracing ──────────────────────────────────────────────────────────
try:
    from langfuse.decorators import observe, langfuse_context
    from langfuse import Langfuse as _LangfuseClient
    _lf = _LangfuseClient(
        public_key      = os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key      = os.getenv("LANGFUSE_SECRET_KEY"),
        host            = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        flush_at        = 1,    # send every event immediately, don't wait to batch
        flush_interval  = 0.1, # flush background queue every 100ms
    )
    _LANGFUSE_ENABLED = True
except ImportError:
    def observe(**kwargs):
        def decorator(fn):
            return fn
        return decorator
    langfuse_context = None
    _lf              = None
    _LANGFUSE_ENABLED = False

anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MAX_ITERATIONS = 6  # hard cap on tool-call rounds before forcing a final answer


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an intelligent document research assistant. "
    "Answer questions EXCLUSIVELY from the content of uploaded documents — never from general knowledge.\n\n"
    "You have three tools:\n"
    "  • search(query, source?)    — find relevant chunks; call multiple times for complex questions\n"
    "  • list_documents()          — see what documents are available before searching\n"
    "  • get_section(source, sec)  — retrieve a complete section when top chunks are not enough\n\n"
    "Strategy:\n"
    "  1. Simple questions: one focused search, then answer.\n"
    "  2. Comparisons: search each document separately with targeted queries.\n"
    "  3. Multi-part questions: decompose into sub-queries, search for each part.\n"
    "  4. Weak first results: retry with different terms, synonyms, or narrower scope.\n"
    "  5. Always cite [Source: filename — Section: name] in your final answer.\n"
    "  6. If after thorough searching you cannot find relevant information, say so clearly."
)


# ── Tool schemas ──────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search",
        "description": (
            "Search uploaded documents for information relevant to a query. "
            "Call multiple times with different, targeted queries to gather all the pieces you need. "
            "Optionally restrict to one document with 'source'. "
            "Returns the most relevant chunks with their source and section labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific, focused search query. More targeted = better results.",
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Optional: restrict to a specific document filename (e.g. 'resume.pdf'). "
                        "Omit to search across all documents."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_documents",
        "description": (
            "List all documents currently in the knowledge base. "
            "Call this first when you need to know what is available, "
            "especially before doing targeted per-document comparisons."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_section",
        "description": (
            "Retrieve ALL chunks from a specific named section of a document. "
            "Use this when search results mention a section you need in full, "
            "or when you need complete coverage of an important section. "
            "The section name must match exactly what appears in search result labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Document filename, e.g. 'annual_report.pdf'",
                },
                "section": {
                    "type": "string",
                    "description": (
                        "Exact section name as it appears in search result labels, "
                        "e.g. 'Employment History' or 'Executive Summary'."
                    ),
                },
            },
            "required": ["source", "section"],
        },
    },
]


# ── Tool execution ────────────────────────────────────────────────────────────

@observe(name="tool-execute", as_type="span")
def _execute_tool(name: str, inputs: dict) -> dict:
    """Dispatch a tool call and return a JSON-serialisable result dict."""

    # Label this span with the actual tool name and its inputs
    if _LANGFUSE_ENABLED and langfuse_context:
        langfuse_context.update_current_observation(
            name=f"tool:{name}",
            input=inputs,
            metadata={"tool_name": name},
        )

    if name == "search":
        query  = inputs["query"]
        source = inputs.get("source") or None
        chunks = retrieve_chunks(query, source_filter=source)
        result = {
            "count": len(chunks),
            "chunks": [
                {
                    "text"      : c["text"],
                    "source"    : c["source"],
                    "section"   : c.get("section", ""),
                    "similarity": c["similarity"],
                    "match_type": c.get("match_type", "vector"),
                }
                for c in chunks
            ],
        }

    elif name == "list_documents":
        docs   = list_documents()
        result = {"documents": docs, "count": len(docs)}

    elif name == "get_section":
        source  = inputs["source"]
        section = inputs["section"]

        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="source",  match=MatchValue(value=source)),
                    FieldCondition(key="section", match=MatchValue(value=section)),
                ]
            ),
            limit=50,
            with_payload=True,
            with_vectors=False,
        )

        chunks = sorted(
            [
                {
                    "text"       : p.payload["text"],
                    "source"     : p.payload["source"],
                    "section"    : p.payload.get("section", ""),
                    "chunk_index": p.payload.get("chunk_index", 0),
                }
                for p in points
            ],
            key=lambda x: x["chunk_index"],
        )
        result = {"count": len(chunks), "chunks": chunks}

    else:
        result = {"error": f"Unknown tool: {name}"}

    # Record the output on the span so it appears in the Langfuse waterfall
    if _LANGFUSE_ENABLED and langfuse_context:
        # Log a lightweight summary (not the full chunk payloads) to keep traces readable
        summary = _result_summary(name, result)
        langfuse_context.update_current_observation(
            output={"summary": summary, "count": result.get("count", 0)},
        )
    return result


def _result_summary(name: str, result: dict) -> str:
    """One-line human-readable summary of a tool result (shown in streaming UI)."""
    if name == "search":
        n       = result.get("count", 0)
        sources = sorted({c["source"] for c in result.get("chunks", [])})
        return f"Found {n} chunks from: {', '.join(sources) or 'none'}"
    if name == "list_documents":
        docs = result.get("documents", [])
        return f"{result.get('count', 0)} document(s): {', '.join(docs)}"
    if name == "get_section":
        return f"Retrieved {result.get('count', 0)} chunks"
    return ""


def _serialize_content(content) -> list[dict]:
    """
    Convert Anthropic SDK content blocks (objects) to plain dicts
    so they can be stored in the messages list and re-sent to the API.
    """
    out = []
    for block in content:
        if block.type == "text":
            out.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            out.append({
                "type" : "tool_use",
                "id"   : block.id,
                "name" : block.name,
                "input": block.input,
            })
    return out


# ── Non-streaming agent (used by evals + /query/agent) ───────────────────────

@observe(name="agent-rag")
def answer_agent(
    question      : str,
    source_filter : str | None       = None,
    history       : list[dict] | None = None,
    max_iterations: int               = MAX_ITERATIONS,
) -> dict:
    """
    Run the full agentic loop and return:
      {
        "answer"    : str,
        "tool_calls": list of {tool, input, summary},
        "iterations": int,
        "sources"   : list of str,
        "off_topic" : False,
        "trace_id"  : "<str>",       # Langfuse trace ID for linking RAGAS scores
      }
    """
    # Tag the trace with metadata visible in the Langfuse UI
    if _LANGFUSE_ENABLED and langfuse_context:
        langfuse_context.update_current_trace(
            input=question,
            tags=["agent", "rag"],
            metadata={
                "source_filter" : source_filter,
                "max_iterations": max_iterations,
            },
        )

    messages: list[dict] = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = question
    if source_filter:
        user_content += f"\n\n(Restrict your searches to the document: {source_filter})"
    messages.append({"role": "user", "content": user_content})

    tool_calls_log: list[dict] = []
    all_sources:    set[str]   = set()

    for iteration in range(max_iterations):
        is_last = (iteration == max_iterations - 1)

        # On the final iteration, remove tools so Claude must answer in text
        kwargs: dict = dict(
            model      = "claude-sonnet-4-6",
            max_tokens = 2048,
            system     = [
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}
            ],
            messages = messages,
        )
        if not is_last:
            kwargs["tools"] = TOOLS
        else:
            # Nudge Claude to write a final answer with what it has
            messages.append({
                "role"   : "user",
                "content": "Based on all the information gathered, please write your comprehensive final answer now.",
            })

        response = anthropic_client.messages.create(**kwargs)

        # ── Text answer ───────────────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            answer = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            trace_id = None
            if _LANGFUSE_ENABLED and langfuse_context:
                trace_id = langfuse_context.get_current_trace_id()
                langfuse_context.update_current_trace(
                    output=answer,
                    metadata={
                        "iterations"      : iteration + 1,
                        "tool_calls_count": len(tool_calls_log),
                        "sources"         : sorted(all_sources),
                        "hit_iteration_cap": False,
                    },
                )
            return {
                "answer"    : answer,
                "tool_calls": tool_calls_log,
                "iterations": iteration + 1,
                "sources"   : sorted(all_sources),
                "off_topic" : False,
                "trace_id"  : trace_id,
            }

        # ── Tool calls ────────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({
                "role"   : "assistant",
                "content": _serialize_content(response.content),
            })
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                result = _execute_tool(block.name, block.input)

                for c in result.get("chunks", []):
                    if c.get("source"):
                        all_sources.add(c["source"])

                summary = _result_summary(block.name, result)
                tool_calls_log.append({
                    "tool"   : block.name,
                    "input"  : block.input,
                    "summary": summary,
                })
                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": block.id,
                    "content"    : json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

    # Fallback — hit MAX_ITERATIONS without an end_turn; flag it in Langfuse
    trace_id = None
    if _LANGFUSE_ENABLED and langfuse_context:
        trace_id = langfuse_context.get_current_trace_id()
        langfuse_context.update_current_trace(
            output="Hit iteration cap — no final answer produced.",
            tags=["agent", "rag", "iteration-cap-hit"],   # easy to filter in UI
            metadata={
                "iterations"      : max_iterations,
                "tool_calls_count": len(tool_calls_log),
                "sources"         : sorted(all_sources),
                "hit_iteration_cap": True,
            },
        )
    return {
        "answer"    : "I reached my search limit. Please try rephrasing your question.",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations,
        "sources"   : sorted(all_sources),
        "off_topic" : False,
        "trace_id"  : trace_id,
    }


# ── Streaming agent (used by /query/agent/stream) ─────────────────────────────

def stream_agent(
    question      : str,
    source_filter : str | None       = None,
    history       : list[dict] | None = None,
    max_iterations: int               = MAX_ITERATIONS,
):
    """
    Generator yielding newline-delimited JSON lines:

      {"type": "agent_start"}
      {"type": "tool_call",   "tool": "search", "input": {...}, "iteration": 1}
      {"type": "tool_result", "tool": "search", "summary": "Found 8 chunks from: doc.pdf"}
      {"type": "token",       "text": "The revenue..."}   ← repeated
      {"type": "done",        "tool_calls": [...], "iterations": 2, "sources": [...]}

    The frontend can use tool_call / tool_result events to show a live
    "🔍 Searching for X..." indicator while the agent is working.
    """
    def emit(obj: dict) -> str:
        return json.dumps(obj) + "\n"

    # @observe doesn't work on generators — use low-level Langfuse API instead
    trace = None
    if _LANGFUSE_ENABLED and _lf:
        trace = _lf.trace(
            name    = "agent-rag-stream",
            input   = question,
            tags    = ["agent", "rag", "stream"],
            metadata= {"source_filter": source_filter, "max_iterations": max_iterations},
        )

    yield emit({"type": "agent_start"})

    messages: list[dict] = []
    for turn in (history or [])[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = question
    if source_filter:
        user_content += f"\n\n(Restrict your searches to the document: {source_filter})"
    messages.append({"role": "user", "content": user_content})

    tool_calls_log: list[dict] = []
    all_sources:    set[str]   = set()

    for iteration in range(max_iterations):
        is_last = (iteration == max_iterations - 1)

        kwargs: dict = dict(
            model      = "claude-sonnet-4-6",
            max_tokens = 2048,
            system     = [
                {"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}
            ],
            messages = messages,
        )
        if not is_last:
            kwargs["tools"] = TOOLS
        else:
            messages.append({
                "role"   : "user",
                "content": "Based on all the information gathered, please write your comprehensive final answer now.",
            })

        response = anthropic_client.messages.create(**kwargs)

        # ── Text answer — stream word-by-word ─────────────────────────────────
        if response.stop_reason == "end_turn":
            answer = "".join(
                b.text for b in response.content if hasattr(b, "text")
            )
            words = answer.split(" ")
            for i, word in enumerate(words):
                suffix = " " if i < len(words) - 1 else ""
                yield emit({"type": "token", "text": word + suffix})

            done_payload = {
                "type"      : "done",
                "tool_calls": tool_calls_log,
                "iterations": iteration + 1,
                "sources"   : sorted(all_sources),
            }
            yield emit(done_payload)
            if trace:
                trace.update(
                    output  = answer,
                    metadata= {
                        "iterations"      : iteration + 1,
                        "tool_calls_count": len(tool_calls_log),
                        "sources"         : sorted(all_sources),
                        "hit_iteration_cap": False,
                    },
                )
            return

        # ── Tool calls — yield live feedback ──────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({
                "role"   : "assistant",
                "content": _serialize_content(response.content),
            })
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                yield emit({
                    "type"     : "tool_call",
                    "tool"     : block.name,
                    "input"    : block.input,
                    "iteration": iteration + 1,
                })

                result  = _execute_tool(block.name, block.input)
                summary = _result_summary(block.name, result)

                for c in result.get("chunks", []):
                    if c.get("source"):
                        all_sources.add(c["source"])

                tool_calls_log.append({
                    "tool"   : block.name,
                    "input"  : block.input,
                    "summary": summary,
                })

                yield emit({
                    "type"   : "tool_result",
                    "tool"   : block.name,
                    "summary": summary,
                })

                tool_results.append({
                    "type"       : "tool_result",
                    "tool_use_id": block.id,
                    "content"    : json.dumps(result),
                })

            messages.append({"role": "user", "content": tool_results})

    yield emit({
        "type"      : "done",
        "tool_calls": tool_calls_log,
        "iterations": max_iterations,
        "sources"   : sorted(all_sources),
    })
    if trace:
        trace.update(
            output  = "Hit iteration cap — no final answer produced.",
            tags    = ["agent", "rag", "stream", "iteration-cap-hit"],
            metadata= {
                "iterations"      : max_iterations,
                "tool_calls_count": len(tool_calls_log),
                "sources"         : sorted(all_sources),
                "hit_iteration_cap": True,
            },
        )
