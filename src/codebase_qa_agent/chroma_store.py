"""Persistent local Chroma vector store."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

from codebase_qa_agent.schemas import Chunk

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


COLLECTION_NAME = "codebase_chunks"


def get_client(chroma_path: Path) -> chromadb.ClientAPI:
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.ClientAPI) -> Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(client: chromadb.ClientAPI) -> Collection:
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    return get_or_create_collection(client)


def upsert_chunks(
    collection: Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    batch_size: int = 64,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=batch_emb,
            documents=[c.text for c in batch],
            metadatas=[c.to_metadata() for c in batch],
        )


def query_vectors(
    collection: Collection,
    query_embedding: list[float],
    *,
    top_k: int = 20,
    where: dict | None = None,
) -> list[tuple[Chunk, float]]:
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)
    hits: list[tuple[Chunk, float]] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
        if doc is None or meta is None:
            continue
        # Chroma cosine distance: lower is better; convert to similarity-like score.
        score = 1.0 / (1.0 + float(dist))
        hits.append((Chunk.from_metadata(doc, chunk_id, meta), score))
    return hits