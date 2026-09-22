# RAGent — Agentic Document Assistant

A learning project built step by step toward answering questions about documents
using retrieval augmented generation (RAG), agent tools, and source citations.

## Current status

Milestone 1: UTF-8 text loading and chunking, plus a working FastAPI health endpoint.
Embeddings, retrieval, PDF support, and agents are not implemented yet.

## Requirements

- Python 3.12 (the version used for this milestone)
- Git

## Run locally

```bash
git clone git@github.com:Oussama-ay/agentic-rag-assistant.git
cd agentic-rag-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

No API key or `.env` file is required for this milestone. `.env.example` holds
placeholders for future integrations; the app does not load it yet.

Open the interactive API documentation at http://127.0.0.1:8000/docs.

In another terminal, check the application:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected: HTTP `200 OK` with this JSON body:

```json
{"status":"ok"}
```

This endpoint confirms the API is responding. It does not check any model,
database, or retrieval service.

## Project structure

```text
app/
  __init__.py       Python package marker
  main.py           FastAPI application and health route
  rag/
    loader.py       Loads a UTF-8 .txt file into a Document
    splitter.py     Splits a Document and preserves its metadata
    preview.py      Command-line chunk inspection
examples/documents/ Small tracked sample documents
tests/             Document loading and chunking checks
data/
  .gitkeep          Keeps the empty data directory in Git
.env.example       Future configuration placeholders, without secrets
.gitignore         Excludes local environments, secrets, and uploaded data
requirements.txt   Direct application dependencies
```

Additional modules will be introduced as their milestones need them.

## Dependencies

- **FastAPI** defines HTTP endpoints and generates interactive API documentation.
- **Uvicorn** runs the application and listens for HTTP requests.
- **langchain-core** provides `Document`, a container for text (`page_content`)
  and source information (`metadata`).
- **langchain-text-splitters** provides the recursive character splitter.

Their supporting packages are installed automatically by pip. This setup uses
Python's built-in `venv` to keep project packages separate from system Python.
See the [FastAPI environment guide](https://fastapi.tiangolo.com/virtual-environments/).

## Inspect document chunks

With the virtual environment activated, run from the repository root:

```bash
python -m app.rag.preview examples/documents/rag_intro.txt
```

You can also place your own UTF-8 `.txt` file in `data/` (ignored by Git):

```bash
python -m app.rag.preview data/notes.txt --chunk-size 300 --chunk-overlap 60
```

The flow is explicit: `load_text(path)` reads one file into a `Document`, then
`split_document(document)` produces a list of smaller `Document` objects.
The preview prints each chunk's text, character count, and metadata.
It runs locally without API keys or model calls.

The default chunk size is **500 characters**, with a target overlap of **100
characters**. These are learning defaults, not tuned retrieval settings. The
recursive splitter tries paragraph boundaries, then lines, spaces, and finally
individual characters. Actual overlap can be smaller, including zero, because
the splitter respects these boundaries. These sizes count characters, not tokens.
See the [LangChain splitter reference](https://reference.langchain.com/python/langchain-text-splitters/character).

Each chunk retains `source` (filename) and includes a zero-based `chunk_index`
and `start_index` (character offset in the original text). Plain text files have
no page numbers. Filenames are simple labels at this stage, not unique document
IDs; distinct files can share the same filename.

The loader rejects unsupported extensions and blank files. Missing files and
invalid UTF-8 also fail with an error. PDF parsing comes in a later milestone.

## Verification

```bash
python -m unittest discover -s tests -v
```

These checks use Python's built-in `unittest`, with no extra test dependency.
They cover Unicode loading, invalid files, chunk size and overlap, source
metadata, character offsets, short documents, and invalid chunk settings.

## Roadmap

- [x] Project setup and health endpoint
- [x] Document loading and chunking (plain text)
- [ ] Embeddings and vector storage
- [ ] Retrieval
- [ ] Grounded answers with sources
- [ ] Retrieval tools and agent orchestration
- [ ] Subagent analysis
- [ ] Document upload and question API endpoints
- [ ] Retrieval and API tests (document processing tests included)
- [ ] Docker packaging
- [ ] Architecture documentation and a reproducible demo
- [ ] Optional frontend
