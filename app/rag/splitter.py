from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_document(
    document: Document, *, chunk_size: int = 500, chunk_overlap: int = 100
) -> list[Document]:
    """Split text by character count, retaining source metadata on each chunk."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents([document])
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks
