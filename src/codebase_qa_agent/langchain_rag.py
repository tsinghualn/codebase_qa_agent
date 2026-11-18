"""LangChain wrappers for embeddings and Chroma (local persistent store)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from codebase_qa_agent.chroma_store import COLLECTION_NAME
from codebase_qa_agent.schemas import Chunk

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def chunks_to_documents(chunks: list[Chunk]) -> list[Document]:
    docs: list[Document] = []
    for chunk in chunks:
        meta = chunk.to_metadata()
        meta["chunk_id"] = chunk.chunk_id
        docs.append(Document(page_content=chunk.text, metadata=meta, id=chunk.chunk_id))
    return docs


def document_to_chunk(doc: Document, chunk_id: str | None = None) -> Chunk:
    meta = doc.metadata or {}
    cid = chunk_id or str(meta.get("chunk_id") or "")
    return Chunk(
        chunk_id=cid,
        path=str(meta.get("path", "")),
        start_line=int(meta.get("start_line", 1)),
        end_line=int(meta.get("end_line", 1)),
        symbol=str(meta.get("symbol") or ""),
        chunk_type=meta.get("chunk_type", "code"),  # type: ignore[arg-type]
        text=doc.page_content,
    )


def _chromadb_client(chroma_path: Path) -> chromadb.ClientAPI:
    chroma_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(chroma_path),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def reset_vector_collection(chroma_path: Path) -> None:
    client = _chromadb_client(chroma_path)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass


def ingest_into_chroma(
    chroma_path: Path,
    chunks: list[Chunk],
    embedding_model_name: str,
) -> Chroma:
    reset_vector_collection(chroma_path)
    embeddings = get_embeddings(embedding_model_name)
    documents = chunks_to_documents(chunks)
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_path),
        ids=[c.chunk_id for c in chunks],
    )
    return store


def open_chroma_vectorstore(chroma_path: Path, embedding_model_name: str) -> Chroma:
    embeddings = get_embeddings(embedding_model_name)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )


def vector_search(
    store: Chroma,
    query: str,
    *,
    top_k: int = 20,
    chunk_type: str | None = None,
) -> list[tuple[Chunk, float]]:
    kwargs: dict = {"k": top_k}
    if chunk_type:
        kwargs["filter"] = {"chunk_type": chunk_type}
    pairs = store.similarity_search_with_score(query, **kwargs)
    hits: list[tuple[Chunk, float]] = []
    for doc, score in pairs:
        cid = str((doc.metadata or {}).get("chunk_id", ""))
        chunk = document_to_chunk(doc, chunk_id=cid)
        # Chroma returns distance (lower is better); map to a ascending relevance score.
        relevance = 1.0 / (1.0 + float(score))
        hits.append((chunk, relevance))
    return hits