import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.rag.vector_store import index_chunks


class RecordingEmbeddings(Embeddings):
    """Fixed vectors for storage tests, not a semantic embedding model."""

    def __init__(self):
        self.texts = []

    def embed_documents(self, texts):
        self.texts = texts
        return [[float(index + 1), 1.0] for index in range(len(texts))]

    def embed_query(self, text):
        raise AssertionError("Indexing should not embed a query.")


class VectorStoreTests(unittest.TestCase):
    def test_index_preserves_chunks_and_pairs_them_with_vectors(self):
        chunks = [
            Document(page_content="First passage", metadata={"source": "a.txt", "chunk_index": 0}),
            Document(page_content="Second passage", metadata={"source": "a.txt", "chunk_index": 1}),
        ]
        embeddings = RecordingEmbeddings()
        store = index_chunks(chunks, embeddings)
        self.assertEqual(embeddings.texts, [chunk.page_content for chunk in chunks])
        records = list(store.store.values())
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["id"] for record in records}), 2)
        for index, (record, chunk) in enumerate(zip(records, chunks)):
            self.assertEqual(record["text"], chunk.page_content)
            self.assertEqual(record["metadata"], chunk.metadata)
            self.assertEqual(record["vector"], [float(index + 1), 1.0])

    def test_invalid_chunks_are_rejected_before_embedding(self):
        for chunks in [[], [Document(page_content=" \n")]]:
            with self.subTest(chunks=chunks):
                embeddings = RecordingEmbeddings()
                with self.assertRaises(ValueError):
                    index_chunks(chunks, embeddings)
                self.assertEqual(embeddings.texts, [])


if __name__ == "__main__":
    unittest.main()
