"""Quick CLI to test hybrid retrieval."""

from __future__ import annotations

import argparse

from codebase_qa_agent.retrieval import RetrievalEngine
from codebase_qa_agent.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=("hybrid", "vector"), default=None)
    args = parser.parse_args(argv)
    settings = get_settings()
    mode = args.mode or settings.retrieval_mode
    engine = RetrievalEngine(settings.chroma_path, settings.embedding_model, mode=mode)
    hits = engine.search(args.question, top_k=args.top_k)
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        print(
            f"{i}. [{hit.source} {hit.score:.4f}] {c.path}:{c.start_line}-{c.end_line} "
            f"({c.symbol})"
        )
        preview = c.text.strip().replace("\n", " ")[:160]
        print(f"   {preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())