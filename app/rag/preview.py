import argparse

from app.rag.loader import load_text
from app.rag.splitter import split_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a text document's chunks.")
    parser.add_argument("path", help="Path to a UTF-8 .txt file")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    try:
        document = load_text(args.path)
        chunks = split_document(
            document, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Loaded {document.metadata['source']}: {len(document.page_content)} characters")
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks:
        print(f"\n--- {chunk.metadata} | {len(chunk.page_content)} characters ---")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
