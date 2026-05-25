"""
Query Pipeline — Embed → Retrieve → Re-rank → Generate

Features:
  - Conversation memory: prior chat turns passed as Claude message history
  - Multi-document synthesis: prompt adapts when chunks span multiple sources
  - Re-ranking: BM25 re-scores initial candidates before generation (no extra deps)
  - Guardrails: off-topic questions rejected when similarity < GUARDRAIL_SIMILARITY
  - Streaming: stream_question() yields ndjson lines for real-time token delivery
"""

import json
import math
import os
from collections import Counter
from openai import OpenAI
import anthropic
from qdrant_client.models import Filter, FieldCondition, MatchValue
from db import qdrant, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

RETRIEVE_K           = 10   # how many chunks to pull from vector DB before re-ranking
TOP_K                = 5    # how many chunks to send to Claude after re-ranking
FALLBACK_THRESHOLD   = 0.35 # if best similarity < this, also run keyword search
GUARDRAIL_SIMILARITY = 0.30 # if best chunk similarity < this, question is off-topic

openai_client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_query(question: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    )
    return response.data[0].embedding


# ── Vector retrieval ─────────────────────────────────────────────────────────

def vector_search(question: str, source_filter: str | None = None, limit: int = RETRIEVE_K) -> list[dict]:
    query_embedding = embed_query(question)

    query_filter = None
    if source_filter:
        query_filter = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
        )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )

    return [
        {
            "text"        : r.payload["text"],
            "source"      : r.payload["source"],
            "chunk_index" : r.payload["chunk_index"],
            "section"     : r.payload.get("section", ""),
            "similarity"  : round(r.score, 4),
            "match_type"  : "vector",
        }
        for r in results
    ]


# ── Keyword fallback ─────────────────────────────────────────────────────────

def keyword_search(question: str, source_filter: str | None = None, top_k: int = 3) -> list[dict]:
    keywords = [w.lower() for w in question.split() if len(w) > 2]

    all_points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )

    scored = []
    for point in all_points:
        payload = point.payload or {}
        if source_filter and payload.get("source") != source_filter:
            continue

        chunk_text = payload.get("text", "").lower()
        hits = sum(1 for kw in keywords if kw in chunk_text)
        if hits > 0:
            scored.append((hits, payload))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "text"        : p["text"],
            "source"      : p["source"],
            "chunk_index" : p["chunk_index"],
            "section"     : p.get("section", ""),
            "similarity"  : round(hits / max(len(keywords), 1), 4),
            "match_type"  : "keyword",
        }
        for hits, p in scored[:top_k]
    ]


# ── Re-ranking (BM25) ────────────────────────────────────────────────────────

def rerank_chunks(question: str, chunks: list[dict]) -> list[dict]:
    """
    BM25 re-ranker — no extra dependencies, pure stdlib math + collections.

    Why BM25 over simple keyword count:
      - Term frequency is length-normalised (long chunks don't win by default)
      - IDF down-weights common words that appear in most chunks
      - k1/b are standard Okapi BM25 tuning parameters (k1=1.5, b=0.75)

    After vector retrieval gives us RETRIEVE_K candidates, BM25 re-ranks them
    by lexical relevance to the question and we keep the top TOP_K.
    """
    if len(chunks) <= 1:
        return chunks[:TOP_K]

    k1, b = 1.5, 0.75
    q_terms = question.lower().split()

    corpus  = [c["text"].lower().split() for c in chunks]
    N       = len(corpus)
    avgdl   = sum(len(doc) for doc in corpus) / N

    def idf(term: str) -> float:
        df = sum(1 for doc in corpus if term in doc)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    idf_cache = {t: idf(t) for t in set(q_terms)}

    for chunk, doc_tokens in zip(chunks, corpus):
        tf_map = Counter(doc_tokens)
        dl     = len(doc_tokens)
        score  = 0.0
        for term in q_terms:
            tf     = tf_map.get(term, 0)
            score += idf_cache[term] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        chunk["rerank_score"] = round(score, 4)

    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:TOP_K]


# ── Combined retrieval ───────────────────────────────────────────────────────

CHUNKS_PER_DOC = 4   # when doing parallel per-doc retrieval, take this many per document


def _all_document_names() -> list[str]:
    from ingest import list_documents
    return list_documents()


def retrieve_chunks(question: str, source_filter: str | None = None) -> list[dict]:
    """
    Single-document or filtered mode: standard single-pass retrieval.
    All-documents mode: parallel per-document retrieval so every uploaded
    document is guaranteed to contribute chunks — prevents one document
    from dominating the results and drowning out the others.
    """
    if source_filter:
        # User pinned a specific document — single-pass is correct
        chunks = vector_search(question, source_filter, limit=RETRIEVE_K)
        best_score = chunks[0]["similarity"] if chunks else 0.0
        if best_score < FALLBACK_THRESHOLD:
            kw_chunks = keyword_search(question, source_filter)
            seen = {(c["source"], c["chunk_index"]) for c in chunks}
            for kc in kw_chunks:
                if (kc["source"], kc["chunk_index"]) not in seen:
                    chunks.append(kc)
                    seen.add((kc["source"], kc["chunk_index"]))
        return rerank_chunks(question, chunks)

    # No filter — search each document separately and merge
    doc_names = _all_document_names()

    if len(doc_names) <= 1:
        # Only one document exists, no need for parallel retrieval
        chunks = vector_search(question, source_filter=None, limit=RETRIEVE_K)
        best_score = chunks[0]["similarity"] if chunks else 0.0
        if best_score < FALLBACK_THRESHOLD:
            kw_chunks = keyword_search(question)
            seen = {(c["source"], c["chunk_index"]) for c in chunks}
            for kc in kw_chunks:
                if (kc["source"], kc["chunk_index"]) not in seen:
                    chunks.append(kc)
                    seen.add((kc["source"], kc["chunk_index"]))
        return rerank_chunks(question, chunks)

    # Multiple documents — retrieve CHUNKS_PER_DOC from each, then merge + re-rank
    merged: list[dict] = []
    seen: set[tuple] = set()

    for doc in doc_names:
        doc_chunks = vector_search(question, source_filter=doc, limit=CHUNKS_PER_DOC)

        # Keyword fallback per document if similarity is low
        best = doc_chunks[0]["similarity"] if doc_chunks else 0.0
        if best < FALLBACK_THRESHOLD:
            kw = keyword_search(question, source_filter=doc, top_k=2)
            doc_chunks.extend(kw)

        for c in doc_chunks:
            key = (c["source"], c["chunk_index"])
            if key not in seen:
                merged.append(c)
                seen.add(key)

    return rerank_chunks(question, merged)


# ── Generation ───────────────────────────────────────────────────────────────

def _build_prompt(
    question: str,
    chunks: list[dict],
    history: list[dict] | None,
) -> tuple[str, list[dict]]:
    """Shared prompt builder — returns (system_prompt, messages_list)."""
    context_parts = []
    for chunk in chunks:
        section_label = f" — Section: {chunk['section']}" if chunk.get("section") else ""
        context_parts.append(f"[Source: {chunk['source']}{section_label}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    unique_sources = {c["source"] for c in chunks}
    if len(unique_sources) > 1:
        synthesis_instruction = (
            "The context comes from MULTIPLE documents. "
            "When answering, synthesize information across all of them and explicitly note "
            "which document each piece of information comes from. "
            "If the documents contain conflicting information, highlight the difference."
        )
    else:
        synthesis_instruction = "Always cite which document and section your answer comes from."

    system_prompt = (
        "You are a helpful assistant that answers questions ONLY based on the provided document context. "
        "Each context block is labelled with its source document and section. "
        "IMPORTANT: If the question cannot be answered from the context provided, respond with exactly: "
        "'I can only answer questions about the uploaded documents.' — do not use general knowledge. "
        f"{synthesis_instruction} "
        "Keep answers concise and directly relevant to the question."
    )

    user_message = (
        f"Here is the relevant context from the uploaded documents:\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer based only on the context above."
    )

    messages: list[dict] = []
    for turn in (history or [])[-10:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    return system_prompt, messages


def generate_answer(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
) -> str:
    if not chunks:
        return "I could not find any relevant information in the uploaded documents to answer your question."

    system_prompt, messages = _build_prompt(question, chunks, history)

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    return response.content[0].text


def _stream_tokens(
    question: str,
    chunks: list[dict],
    history: list[dict] | None = None,
):
    """Generator that yields raw text tokens from Claude's streaming API."""
    if not chunks:
        yield "I could not find any relevant information in the uploaded documents to answer your question."
        return

    system_prompt, messages = _build_prompt(question, chunks, history)

    with anthropic_client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ── Public entry points ──────────────────────────────────────────────────────

_OFF_TOPIC_MSG = (
    "I can only answer questions about the uploaded documents. "
    "This question doesn't appear to relate to the content in your documents."
)


def _sources_payload(chunks: list[dict]) -> list[dict]:
    return [
        {
            "source"       : c["source"],
            "chunk_index"  : c["chunk_index"],
            "section"      : c.get("section", ""),
            "similarity"   : c["similarity"],
            "match_type"   : c.get("match_type", "vector"),
            "rerank_score" : round(c.get("rerank_score", 0.0), 4),
        }
        for c in chunks
    ]


def answer_question(
    question: str,
    source_filter: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """End-to-end: question → retrieve → guardrail → generate → return answer + sources."""
    chunks = retrieve_chunks(question, source_filter=source_filter)

    best_similarity = max((c["similarity"] for c in chunks), default=0.0)
    if best_similarity < GUARDRAIL_SIMILARITY:
        return {
            "question" : question,
            "answer"   : _OFF_TOPIC_MSG,
            "multi_doc": False,
            "sources"  : [],
            "off_topic": True,
        }

    answer = generate_answer(question, chunks, history=history)
    unique_sources = {c["source"] for c in chunks}

    return {
        "question" : question,
        "answer"   : answer,
        "multi_doc": len(unique_sources) > 1,
        "sources"  : _sources_payload(chunks),
        "off_topic": False,
    }


def stream_question(
    question: str,
    source_filter: str | None = None,
    history: list[dict] | None = None,
):
    """
    Generator yielding newline-delimited JSON lines:
      {"type": "metadata", "sources": [...], "multi_doc": bool, "off_topic": bool}
      {"type": "token", "text": "..."}   (repeated)
      {"type": "done"}
    """
    chunks = retrieve_chunks(question, source_filter=source_filter)

    best_similarity = max((c["similarity"] for c in chunks), default=0.0)
    if best_similarity < GUARDRAIL_SIMILARITY:
        yield json.dumps({"type": "metadata", "sources": [], "multi_doc": False, "off_topic": True}) + "\n"
        yield json.dumps({"type": "token", "text": _OFF_TOPIC_MSG}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"
        return

    unique_sources = {c["source"] for c in chunks}
    yield json.dumps({
        "type"     : "metadata",
        "sources"  : _sources_payload(chunks),
        "multi_doc": len(unique_sources) > 1,
        "off_topic": False,
    }) + "\n"

    for token in _stream_tokens(question, chunks, history=history):
        yield json.dumps({"type": "token", "text": token}) + "\n"

    yield json.dumps({"type": "done"}) + "\n"
