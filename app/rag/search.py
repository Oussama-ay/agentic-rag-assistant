import argparse

from app.rag.embeddings import LocalEmbeddings
from app.rag.loader import load_text
from app.rag.retriever import retrieve
from app.rag.splitter import split_document
from app.rag.vector_store import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Find relevant passages in a text document.")
    parser.add_argument("path", help="Path to a UTF-8 .txt file")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum results (default: 3)")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("The question must contain non-whitespace text.")
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer.")

    try:
        document = load_text(args.path)
        chunks = split_document(document)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Indexing {len(chunks)} chunks with the local model...", flush=True)
    embeddings = LocalEmbeddings()
    store = index_chunks(chunks, embeddings)
    results = retrieve(args.question, store, top_k=args.top_k)

    print(f"\nQuestion: {args.question}")
    print(f"Retrieved {len(results)} chunks (scores are cosine similarities, not confidence):")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(
            f"\n{rank}. {chunk.metadata['source']} | "
            f"chunk={chunk.metadata['chunk_index']} | "
            f"start={chunk.metadata['start_index']} | score={score:.4f}"
        )
        print(chunk.page_content)


if __name__ == "__main__":
    main()
