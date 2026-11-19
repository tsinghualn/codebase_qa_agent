# Codebase Q&A Agent

LangGraph agent that answers questions about a **pinned** public repo (LangGraph) with **file:line citations**, hybrid RAG, evals, and an MCP server.

**Unofficial demo — not affiliated with LangChain.**

## Milestone status

| Milestone | Status |
|-----------|--------|
| M0 Scaffold + golden set | Done (refine golden answers at your pin) |
| M1 Ingest + retrieval | Done |
| M2 Agent + citations | Done — `make ask` |
| M3 Rerank + eval comparison | `make eval` → `eval-results.md` |
| M4 FastMCP server | `make mcp` (stdio) — see [MCP](#mcp-cursor--other-clients) |

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
make ask Q="How does checkpointing work?"   # cited answer (needs OPENAI_API_KEY)
make eval     # vector vs hybrid vs hybrid+rerank (writes eval-results.md)
make test
make mcp        # FastMCP stdio server (after ingest)
```

### MCP (Cursor & other clients)

The MCP server exposes **`search_code`**, **`search_docs`**, and **`read_file`** over the same hybrid index and pinned corpus as `make query` / `make ask`. No LLM key is required for MCP tools.

1. `make install` and `make ingest` so `data/chroma/` exists.
2. Copy [`.mcp.json.example`](.mcp.json.example) into your Cursor MCP config (or merge the `langgraph-codebase` entry). Adjust the Python path if you do not use `.venv` at the project root.
3. Restart MCP / Cursor. Tools should appear as **langgraph-codebase**.

Local smoke test (server blocks on stdio — Ctrl+C to exit):

```bash
make mcp
# or: codebase-mcp
```

Environment variables (`CHROMA_PATH`, `RETRIEVAL_MODE`, `EMBEDDING_MODEL`, etc.) are read from `.env` via `get_settings()`, same as the CLI.

### Retrieval stack (M3)

Default: **hybrid** (BM25 + vectors + RRF). Cross-encoder rerank is optional (`RERANK_ENABLED=true`); on our LangGraph golden set it **lowered** top-3 file hit rate vs hybrid alone — see [`eval-results.md`](eval-results.md).

### Vector store

**Chroma** is used as a **local persistent** store (`CHROMA_PATH=data/chroma`). No Pinecone or cloud vector DB required for M1. BM25 metadata is stored alongside as `data/chroma/bm25_meta.json`.

Embeddings default to **`sentence-transformers/all-MiniLM-L6-v2`** via **LangChain** `HuggingFaceEmbeddings` → **`langchain-chroma`**. Docs use LangChain `MarkdownHeaderTextSplitter`; Python stays **AST chunking** for precise `file:line` citations. OpenAI is for the agent (M2+), not ingest.

## Golden eval set

Hand-written questions live in `evals/golden.jsonl`. Replace every `VERIFY:` reference answer after you pin `data/pin.json` to a real commit and read the code at that SHA. Include 2–3 unanswerable questions (already seeded).

## Corpus pin

`data/pin.json` records the LangGraph commit SHA and scope (`libs/langgraph`, `docs`). The cloned repo goes to `data/corpus/` (gitignored).

## License

MIT — corpus is Apache-2.0 upstream; do not redistribute the cloned corpus in this repo.