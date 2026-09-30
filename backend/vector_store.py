"""Chroma Cloud vector store for job embeddings.

Connection settings come from environment variables (CHROMA_API_KEY,
CHROMA_TENANT, CHROMA_DATABASE). Nothing is hardcoded; if credentials
are missing the client raises loudly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb

COLLECTION_NAME = "jobs"


@dataclass
class JobMatch:
    """A single Chroma query hit with its metadata and cosine distance."""

    id: str
    distance: float
    metadata: dict
    document: str


def _get_client() -> "chromadb.ClientAPI":
    import chromadb

    api_key = os.environ["CHROMA_API_KEY"]
    tenant = os.environ["CHROMA_TENANT"]
    database = os.environ["CHROMA_DATABASE"]
    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )


_client_cache: "chromadb.ClientAPI | None" = None


def get_client() -> "chromadb.ClientAPI":
    """Return a cached Chroma Cloud client (one connection, reused)."""
    global _client_cache
    if _client_cache is None:
        _client_cache = _get_client()
    return _client_cache


def get_jobs_collection():
    """Get or create the cosine-distance 'jobs' collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_jobs(ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    """Idempotent upsert of job documents into the collection."""
    collection = get_jobs_collection()
    batch = 100
    for start in range(0, len(ids), batch):
        collection.upsert(
            ids=ids[start : start + batch],
            documents=documents[start : start + batch],
            embeddings=embeddings[start : start + batch],
            metadatas=metadatas[start : start + batch],
        )


def query_jobs(embedding: list[float], top_k: int = 50) -> list[JobMatch]:
    """Query the collection by embedding, returning matches with metadata."""
    collection = get_jobs_collection()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count() or top_k),
        include=["distances", "metadatas", "documents"],
    )
    ids = result["ids"][0]
    distances = result["distances"][0]
    metadatas = result["metadatas"][0]
    documents = result["documents"][0]
    return [
        JobMatch(
            id=job_id,
            distance=distance,
            metadata=metadata or {},
            document=document or "",
        )
        for job_id, distance, metadata, document in zip(ids, distances, metadatas, documents)
    ]
