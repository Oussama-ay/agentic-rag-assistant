from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


def retrieve(
    query: str, store: InMemoryVectorStore, *, top_k: int = 3
) -> list[tuple[Document, float]]:
    """Return up to top_k chunks, ordered by descending cosine similarity."""
    if not query.strip():
        raise ValueError("The question must contain non-whitespace text.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    return store.similarity_search_with_score(query.strip(), k=top_k)
