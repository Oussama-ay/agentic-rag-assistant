import math
import os
import unittest

from langchain_core.documents import Document

from app.rag.vector_store import index_chunks


@unittest.skipUnless(
    os.environ.get("RUN_LOCAL_EMBEDDINGS") == "1",
    "Set RUN_LOCAL_EMBEDDINGS=1 to run the real model (first run downloads it).",
)
class LocalEmbeddingTests(unittest.TestCase):
    def test_real_model_vectors_and_storage(self):
        from app.rag.embeddings import LocalEmbeddings

        embeddings = LocalEmbeddings()
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


if __name__ == "__main__":
    unittest.main()
