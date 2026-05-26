"""
Phase 1 - Ingestion Pipeline
Load documents → section-aware chunk → embed → store in Qdrant

FIX: Section-aware chunking splits the document on headers (company names,
section titles) BEFORE applying character-based chunking. This prevents
content from one company/section bleeding into the next chunk.
"""

import os
import re
import uuid
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from db import qdrant, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "uploads")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Fine-grained splitter used WITHIN each section
_fine_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)

# ── Section-header patterns ──────────────────────────────────────────────────
# Matches lines like:
#   "Senior Product Manager at Hindustan Times, Delhi"
#   "Employment History"   "Education"   "Skills"   "Personal Projects"
#   "MBA, Fore School of Management, Delhi"
#   "Intern at EY, Delhi"
#   "Rewards & Recognition"

_SECTION_PATTERNS = [
    r"^\s*(Employment History|Education|Skills|Personal Projects|"
    r"Internships?|Rewards?\s*&?\s*Recognition|Languages?|Profile|"
    r"Certifications?|Summary|Experience|Projects?|Publications?|"
    r"Volunteer|Achievements?|Awards?)\s*$",

    # Role at Company patterns
    r"^\s*[\w\s/,]+\bat\b[\w\s,]+$",

    # Degree, School patterns  e.g. "MBA, Fore School of Management, Delhi"
    r"^\s*(MBA|BTech|B\.Tech|BCA|MCA|MSc|BSc|PhD|B\.E\.|M\.E\.)[\w\s,\.]+$",
]

_SECTION_RE = re.compile(
    "|".join(_SECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(file_path: str) -> str:
    """Extract raw text from PDF, DOCX, or TXT files."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)

    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split raw text into (section_title, section_body) pairs using header detection.
    Falls back to treating the whole document as one section if no headers found.
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        if _SECTION_RE.match(line) and len(line.strip()) > 3:
            # Save the previous section
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_title, body))

    return sections if sections else [("Document", text)]


def section_aware_chunks(text: str) -> list[dict]:
    """
    1. Split document into sections on headers
    2. Fine-chunk each section independently
    3. Tag every chunk with its section name

    Returns list of {text, section} dicts.
    """
    sections = split_into_sections(text)
    all_chunks: list[dict] = []

    for title, body in sections:
        # Prepend the section title to every chunk so the embedding
        # carries the context ("Ziploan | designed customer journey...")
        prefixed = f"{title}\n{body}"
        sub_chunks = _fine_splitter.split_text(prefixed)
        for chunk in sub_chunks:
            all_chunks.append({"text": chunk, "section": title})

    return all_chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks using OpenAI embeddings."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


def ingest_document(file_path: str, filename: str) -> dict:
    """
    Full ingestion pipeline for one document:
    1. Extract text
    2. Section-aware split
    3. Embed chunks
    4. Store in Qdrant (with section metadata)
    """
    raw_text = extract_text(file_path)
    if not raw_text.strip():
        raise ValueError("Document appears to be empty or unreadable.")

    chunk_dicts = section_aware_chunks(raw_text)
    if not chunk_dicts:
        raise ValueError("No chunks produced from document.")

    texts = [c["text"] for c in chunk_dicts]

    # Embed in batches of 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), 100):
        all_embeddings.extend(embed_texts(texts[i : i + 100]))

    # Delete old chunks for this file
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filename))]
        ),
    )

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "source": filename,
                "chunk_index": i,
                "section": chunk_dicts[i]["section"],
                "text": chunk_dicts[i]["text"],
            },
        )
        for i, embedding in enumerate(all_embeddings)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    return {
        "filename": filename,
        "chunks": len(points),
        "characters": len(raw_text),
        "sections": len(set(c["section"] for c in chunk_dicts)),
    }


def list_documents() -> list[str]:
    """Return all unique source document names stored in Qdrant."""
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    sources = {p.payload["source"] for p in results if p.payload}
    return sorted(sources)


def delete_document(filename: str) -> bool:
    """Remove all chunks for a given document from Qdrant."""
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filename))]
        ),
    )
    return True
