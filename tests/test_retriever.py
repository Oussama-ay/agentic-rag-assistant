import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from app.rag.retriever import retrieve
from app.rag.vector_store import index_chunks


class GeometryEmbeddings(Embeddings):
    """Known vectors let tests verify ranking without downloading a model."""

    def embed_documents(self, texts):
        vectors = {"opposite": [-1.0, 0.0], "related": [1.0, 1.0], "closest": [1.0, 0.0]}
        return [vectors[text] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = GeometryEmbeddings()
        self.chunks = [
            Document(page_content=text, metadata={"source": "test.txt", "chunk_index": index})
            for index, text in enumerate(["opposite", "related", "closest"])
        ]
        self.store = index_chunks(self.chunks, self.embeddings)

    def test_ranking_scores_and_metadata(self):
        results = retrieve("question", self.store, top_k=2)
        self.assertEqual([doc.page_content for doc, _ in results], ["closest", "related"])
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertAlmostEqual(results[1][1], 2 ** -0.5)
        self.assertEqual(results[0][0].metadata, self.chunks[2].metadata)
        self.assertEqual(results[1][0].metadata, self.chunks[1].metadata)

    def test_top_k_larger_than_index(self):
        results = retrieve("question", self.store, top_k=20)
        self.assertEqual(len(results), 3)
        self.assertAlmostEqual(results[-1][1], -1.0)

    def test_empty_store(self):
        store = InMemoryVectorStore(embedding=self.embeddings)
        self.assertEqual(retrieve("question", store), [])

    def test_invalid_query(self):
        for query in ["", " \n\t"]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                retrieve(query, self.store)

    def test_invalid_top_k(self):
        for top_k in [0, -1, 1.5, True]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retrieve("question", self.store, top_k=top_k)


if __name__ == "__main__":
    unittest.main()
