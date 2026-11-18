"""Build Chroma index + BM25 corpus from pinned repo (LangChain embeddings + Chroma)."""

from __future__ import annotations

import argparse
import sys

from codebase_qa_agent.bm25_index import BM25Index
from codebase_qa_agent.chunking import chunk_file
from codebase_qa_agent.corpus import ensure_corpus, iter_corpus_files, load_pin
from codebase_qa_agent.langchain_rag import ingest_into_chroma
from codebase_qa_agent.settings import get_settings


def build_index(*, write_pin: bool = False) -> int:
    settings = get_settings()
    pin = load_pin(settings.project_root)
    corpus_dir = ensure_corpus(settings.project_root, pin, write_pin=write_pin)
    files = iter_corpus_files(
        corpus_dir,
        pin["scope_paths"],
        pin.get("exclude_globs", []),
        int(pin.get("max_file_bytes", 204800)),
    )
    chunks = []
    for path in files:
        relative = path.relative_to(corpus_dir).as_posix()
        chunks.extend(chunk_file(relative, path))

    if not chunks:
        print("No chunks produced — check scope_paths and clone.", file=sys.stderr)
        return 1

    print(f"Chunked {len(files)} files → {len(chunks)} chunks")
    ingest_into_chroma(settings.chroma_path, chunks, settings.embedding_model)
    BM25Index(chunks).save(settings.chroma_path)
    print(f"Index written to {settings.chroma_path} (LangChain HuggingFace → Chroma)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest pinned corpus into Chroma + BM25")
    parser.add_argument(
        "--write-pin",
        action="store_true",
        help="If commit_sha is unset, resolve HEAD and write data/pin.json",
    )
    args = parser.parse_args(argv)
    return build_index(write_pin=args.write_pin)


if __name__ == "__main__":
    raise SystemExit(main())