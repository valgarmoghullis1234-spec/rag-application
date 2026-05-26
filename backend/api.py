"""
FastAPI server — exposes ingestion and query endpoints
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ingest import ingest_document, list_documents, delete_document
from query import answer_question, stream_question
from agent import answer_agent, stream_agent

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Upload endpoint ──────────────────────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not supported. Use PDF, DOCX, or TXT.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_document(save_path, file.filename)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "success", **result}


# ── List documents endpoint ──────────────────────────────────────────────────

@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}


# ── Delete document endpoint ─────────────────────────────────────────────────

@app.delete("/documents/{filename}")
def remove_document(filename: str):
    deleted = delete_document(filename)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"status": "deleted", "filename": filename}


# ── Query endpoint ───────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    source_filter: str | None = None
    history: list[HistoryMessage] = []


@app.post("/query")
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result = answer_question(req.question, source_filter=req.source_filter, history=history)
    return result


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.history]

    return StreamingResponse(
        stream_question(req.question, source_filter=req.source_filter, history=history),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Agentic query endpoints ───────────────────────────────────────────────────

@app.post("/query/agent")
def query_agent(req: QueryRequest):
    """
    Agentic RAG — Claude drives multi-step retrieval with tools.
    Better than /query for comparisons, multi-hop, and self-correcting searches.
    Slower and more expensive; use for complex questions.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.history]
    result  = answer_agent(req.question, source_filter=req.source_filter, history=history)
    return result


@app.post("/query/agent/stream")
def query_agent_stream(req: QueryRequest):
    """
    Streaming agentic RAG.
    Yields ndjson lines:
      {"type": "agent_start"}
      {"type": "tool_call",   "tool": "search",  "input": {...},   "iteration": 1}
      {"type": "tool_result", "tool": "search",  "summary": "..."}
      {"type": "token",       "text": "..."}
      {"type": "done",        "tool_calls": [...], "iterations": N, "sources": [...]}
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    history = [{"role": m.role, "content": m.content} for m in req.history]

    return StreamingResponse(
        stream_agent(req.question, source_filter=req.source_filter, history=history),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
