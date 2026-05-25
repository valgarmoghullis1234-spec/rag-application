"""
Single shared Qdrant client — imported by both ingest.py and query.py
to avoid the 'already accessed by another instance' error.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "qdrant_db")
COLLECTION_NAME = "documents"
EMBEDDING_DIM = 1536  # text-embedding-3-small

qdrant = QdrantClient(path=QDRANT_DIR)

existing = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
