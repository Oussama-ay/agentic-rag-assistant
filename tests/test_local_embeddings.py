import math
import os
from pathlib import Path
import unittest

from langchain_core.documents import Document

from app.rag.vector_store import index_chunks
from app.rag.loader import load_text
from app.rag.retriever import retrieve
from app.rag.splitter import split_document


@unittest.skipUnless(
    os.environ.get("RUN_LOCAL_EMBEDDINGS") == "1",
    "Set RUN_LOCAL_EMBEDDINGS=1 to run the real model (first run downloads it).",
)
class LocalEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.rag.embeddings import LocalEmbeddings

        cls.embeddings = LocalEmbeddings()

    def test_real_model_vectors_and_storage(self):
        embeddings = self.embeddings
        chunks = [
            Document(page_content="Authentication verifies a user's identity.", metadata={"source": "auth.txt"}),
            Document(page_content="Bread dough needs flour, water, and yeast.", metadata={"source": "bread.txt"}),
        ]
        store = index_chunks(chunks, embeddings)
        records = list(store.store.values())
        self.assertEqual(len(records), len(chunks))
        vectors = [record["vector"] for record in records]
        query_vector = embeddings.embed_query("How is a user's identity verified?")
        for vector in vectors + [query_vector]:
            self.assertEqual(len(vector), 384)
            self.assertTrue(all(math.isfinite(value) for value in vector))
            self.assertGreater(sum(value * value for value in vector), 0)
        self.assertNotEqual(vectors[0], vectors[1])
        for record, chunk in zip(records, chunks):
            self.assertEqual(record["text"], chunk.page_content)
            self.assertEqual(record["metadata"], chunk.metadata)

    def test_sample_retrieval(self):
        path = Path(__file__).resolve().parents[1] / "examples/documents/rag_intro.txt"
        chunks = split_document(load_text(path))
        store = index_chunks(chunks, self.embeddings)
        cases = [
            ("Why do chunks overlap?", "Overlap repeats some nearby text"),
            ("How does the system compare a question with stored passages?", "embeds the question"),
            ("What should the assistant do if evidence is insufficient?", "assistant should say so"),
        ]
        for question, expected_passage in cases:
            with self.subTest(question=question):
                results = retrieve(question, store, top_k=2)
                self.assertEqual(len(results), 2)
                self.assertIn(expected_passage, results[0][0].page_content)
                self.assertEqual(results[0][0].metadata["source"], path.name)
                self.assertGreaterEqual(results[0][1], results[1][1])


if __name__ == "__main__":
    unittest.main()
