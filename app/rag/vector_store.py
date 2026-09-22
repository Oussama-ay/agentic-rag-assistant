from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


def index_chunks(
    chunks: list[Document], embeddings: Embeddings
) -> InMemoryVectorStore:
    """Embed chunks and store their vectors, text, and metadata in memory."""
    if not chunks:
        raise ValueError("At least one chunk is required for indexing.")
    if any(not chunk.page_content.strip() for chunk in chunks):
        raise ValueError("Chunks must contain non-whitespace text.")

    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store
