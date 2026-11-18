# Codebase Q&A Agent

LangGraph agent that answers questions about a **pinned** public repo (LangGraph) with **file:line citations**, hybrid RAG, evals, and an MCP server.

**Unofficial demo — not affiliated with LangChain.**

## Milestone status

| Milestone | Status |
|-----------|--------|
| M0 Scaffold + golden set | Done (refine golden answers at your pin) |
| M1 Ingest + retrieval | Chroma + BM25 hybrid |

## Where to put API keys (security)

| Location | Use |
|----------|-----|
| **`.env`** (project root) | **Local development only.** Gitignored. Put real keys here. |
| **`.env.example`** | Committed template — variable **names** only, never real values. |
| **GitHub Actions** | Repo → Settings → Secrets — for CI evals with LLM judge later. |
| **Hugging Face Space** | Space → Settings → Secrets — for deployed Gradio app (M5). |

### Setup

```bash
cp .env.example .env
# Edit .env on your machine only — do not commit it.
```

Supported variables (see `.env.example`):

- `OPENAI_API_KEY` + optional `OPENAI_MODEL`
- `ANTHROPIC_API_KEY` + optional `ANTHROPIC_MODEL`
- `LLM_PROVIDER=openai|anthropic`

**Rules**

- Never commit `.env`, never paste keys into `golden.jsonl`, issues, or README.
- Retrieval-only path (M1) runs **without** any LLM key.
- The browser never receives API keys — only server-side code reads `os.environ` after `load_dotenv()`.

Optional: store secrets in 1Password / macOS Keychain and copy into `.env` when developing; the app only reads environment variables.

## Quick start

```bash
uv venv .venv && source .venv/bin/activate
make install
make ingest   # clone pinned repo, chunk, embed → data/chroma/ (needs ~5–15 min first run)
make query Q="How does checkpointing work?"
make eval     # retrieval metrics on golden set
make test
```

### Vector store

**Chroma** is used as a **local persistent** store (`CHROMA_PATH=data/chroma`). No Pinecone or cloud vector DB required for M1. BM25 metadata is stored alongside as `data/chroma/bm25_meta.json`.

Embeddings default to **`sentence-transformers/all-MiniLM-L6-v2`** via **LangChain** `HuggingFaceEmbeddings` → **`langchain-chroma`**. Docs use LangChain `MarkdownHeaderTextSplitter`; Python stays **AST chunking** for precise `file:line` citations. OpenAI is for the agent (M2+), not ingest.

## Golden eval set

Hand-written questions live in `evals/golden.jsonl`. Replace every `VERIFY:` reference answer after you pin `data/pin.json` to a real commit and read the code at that SHA. Include 2–3 unanswerable questions (already seeded).

## Corpus pin

`data/pin.json` records the LangGraph commit SHA and scope (`libs/langgraph`, `docs`). The cloned repo goes to `data/corpus/` (gitignored).

## License

MIT — corpus is Apache-2.0 upstream; do not redistribute the cloned corpus in this repo.