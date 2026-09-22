import argparse

from app.rag.embeddings import LocalEmbeddings, MODEL_NAME
from app.rag.loader import load_text
from app.rag.splitter import split_document
from app.rag.vector_store import index_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and index a text document locally.")
    parser.add_argument("path", help="Path to a UTF-8 .txt file")
    args = parser.parse_args()

    try:
        document = load_text(args.path)
        chunks = split_document(document)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Loaded {document.metadata['source']}; created {len(chunks)} chunks", flush=True)
    print(f"Loading {MODEL_NAME} (first run downloads the model)...", flush=True)
    embeddings = LocalEmbeddings()
    store = index_chunks(chunks, embeddings)

    for record in store.store.values():
        print(
            f"Chunk {record['metadata']['chunk_index']}: "
            f"source={record['metadata']['source']}, "
            f"characters={len(record['text'])}, "
            f"dimensions={len(record['vector'])}"
        )
        print(f"  First 5 coordinates: {record['vector'][:5]}")
    print(f"Stored {len(store.store)} vectors in memory. The index ends with this process.")


if __name__ == "__main__":
    main()
