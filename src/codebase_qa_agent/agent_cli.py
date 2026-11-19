"""CLI for the M2 cited Q&A agent."""

from __future__ import annotations

import argparse

from codebase_qa_agent.agent import answer_question


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the codebase Q&A agent")
    parser.add_argument("question", help="Question about the pinned corpus")
    args = parser.parse_args(argv)
    question = args.question
    print(answer_question(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())