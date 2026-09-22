# RAGent — Agentic Document Assistant

A learning project built step by step toward answering questions about documents
using retrieval augmented generation (RAG), agent tools, and source citations.

## Current status

Milestone 3: UTF-8 text loading, chunking, local embeddings, in-memory vector
storage, and semantic retrieval, plus a working FastAPI health endpoint.
PDF support, answer generation, and agents are not implemented yet.

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
    embeddings.py   Local embedding model and LangChain adapter
    vector_store.py Embeds and stores chunks in memory
    index.py        Command-line indexing demo
    retriever.py    Top-k similarity search, independent of the API
    search.py       Command-line question and retrieved passages
examples/documents/ Small tracked sample documents
tests/             Document, storage, and retrieval checks
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
- **FastEmbed** runs the embedding model locally through ONNX Runtime. It
  downloads model files on first use; no API key or GPU is required.

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

## Embed and index a document

```bash
python -m pip install -r requirements.txt
python -m app.rag.index examples/documents/rag_intro.txt
```

The explicit pipeline is:

```text
load_text → split_document → LocalEmbeddings → index_chunks
file        chunks           embedding model   in-memory vector store
```

`LocalEmbeddings` uses **BAAI/bge-small-en-v1.5**, an English model that returns
384 numbers per text. This is the embedding dimension: the vector's length,
not the document's character count. These learned coordinates let a later
retrieval step compare passages and questions by similarity.
See the [FastEmbed getting-started guide](https://qdrant.github.io/fastembed/Getting%20Started/).

The adapter implements LangChain's `embed_documents` and `embed_query` methods.
It converts FastEmbed's arrays into ordinary Python lists. `index_chunks` passes
the chunks to `InMemoryVectorStore.add_documents`, which calls `embed_documents`
and keeps each vector with its text, metadata, and generated ID. You can follow
that call in the [LangChain store reference](https://reference.langchain.com/python/langchain-core/vectorstores/in_memory).

The demo prints the stored vector dimensions and first five coordinates for
every chunk, so you can see the output of this stage before adding retrieval.

- The first run requires internet access to download model files. They are
  cached in `.cache/fastembed/`, ignored by Git. Downloads may take several minutes.
- Embedding computation happens on your CPU, using two model threads. Document
  text is not sent to an embedding API. API placeholders in `.env.example` are
  reserved for later milestones and are not needed here.
- Model files persist in the cache, but the **document index lives only in RAM**.
  It is discarded when the process exits; each demo run re-embeds the document.
- The model is intended for English text. Current chunk sizes count characters;
  model inputs are tokenized and have their own length limits. The small default
  chunks are a starting point, not a guarantee for all languages or inputs.

## Retrieve relevant passages

```bash
python -m app.rag.search examples/documents/rag_intro.txt "Why do chunks overlap?" --top-k 2
```

This command indexes the file, embeds the question using the same local model,
and returns up to `top_k` chunks ordered by descending cosine similarity.
The default is three results. Each result prints its text, source filename,
zero-based chunk index, character offset, and score. No answer-generating LLM
is involved.

```text
Question → query embedding → cosine similarity against stored vectors → top-k chunks
```

The reusable function is `retrieve(query, store, top_k=3)`. It returns a list of
`(Document, score)` pairs. Indexing happens separately, so callers can reuse a
store for multiple questions. The command-line demo rebuilds the index each run.

Cosine similarity compares the directions of two vectors. Scores range from
-1 to 1 (subject to floating-point rounding), with larger values indicating
more similar directions. A score is **not a confidence percentage**.
The sample question ranked the passage explaining overlap first, at about 0.775
in a local run; exact scores can vary with the runtime.

`top_k` must be a positive integer and the question must contain text. An empty
store returns no results; requesting more chunks than exist returns all of them.
There is currently no relevance threshold: even an unrelated question can return
chunks from a nonempty store. Retrieval alone does not establish that the evidence
is sufficient to answer a question.

## Verification

```bash
python -m unittest discover -s tests -v
```

These checks use Python's built-in `unittest`, with no extra test dependency.
They cover Unicode loading, invalid files, chunk size and overlap, source
metadata, character offsets, short documents, and invalid chunk settings.
Storage tests also check that vectors stay paired with their chunk text and
metadata and that empty inputs are rejected. Those tests use fixed vectors and
do not download a model or measure semantic quality. Run the indexing demo above
to verify the real model on your machine.

An opt-in integration test checks real document and query vectors for finite,
nonzero, 384-dimensional output and verifies their stored text and metadata:

```bash
RUN_LOCAL_EMBEDDINGS=1 python -m unittest discover -s tests -v
```

These integration tests may download the model on their first run. They also
check three questions about the tracked sample document: overlap, question
embeddings, and insufficient evidence. Each expected passage must rank first.
This small smoke evaluation is not a general retrieval-quality benchmark.

The normal suite checks retrieval ordering against known cosine scores, source
metadata preservation, invalid questions and `top_k`, an empty index, and fewer
available chunks than requested. Its fixed-vector tests require no model download.

## Roadmap

- [x] Project setup and health endpoint
- [x] Document loading and chunking (plain text)
- [x] Local embeddings and in-memory vector storage
- [x] Semantic retrieval with scored source passages
- [ ] Grounded answers with sources
- [ ] Retrieval tools and agent orchestration
- [ ] Subagent analysis
- [ ] Document upload and question API endpoints
- [x] Document, storage, and retrieval tests
- [ ] API tests
- [ ] Docker packaging
- [ ] Architecture documentation and a reproducible demo
- [ ] Optional frontend
