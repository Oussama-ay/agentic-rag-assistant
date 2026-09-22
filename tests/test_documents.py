from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.documents import Document

from app.rag.loader import load_text
from app.rag.splitter import split_document


class DocumentTests(unittest.TestCase):
    def test_load_preserves_unicode_and_filename(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "notes.txt"
            path.write_text("Hello, café!\nSecond line.", encoding="utf-8")
            document = load_text(path)
            self.assertEqual(document.page_content, "Hello, café!\nSecond line.")
            self.assertEqual(document.metadata, {"source": "notes.txt"})

    def test_invalid_documents(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            path.write_text(" \n\t", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_text(path)
            with self.assertRaises(FileNotFoundError):
                load_text(Path(directory) / "missing.txt")
            with self.assertRaises(ValueError):
                load_text(Path(directory) / "file.pdf")
            path.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                load_text(path)

    def test_chunk_boundaries_overlap_and_metadata(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        document = Document(page_content=text, metadata={"source": "letters.txt"})
        chunks = split_document(document, chunk_size=10, chunk_overlap=3)
        self.assertGreater(len(chunks), 1)
        reconstructed = chunks[0].page_content
        for index, chunk in enumerate(chunks):
            self.assertLessEqual(len(chunk.page_content), 10)
            self.assertEqual(chunk.metadata["source"], "letters.txt")
            self.assertEqual(chunk.metadata["chunk_index"], index)
            start = chunk.metadata["start_index"]
            self.assertEqual(text[start:start + len(chunk.page_content)], chunk.page_content)
            if index:
                self.assertEqual(chunks[index - 1].page_content[-3:], chunk.page_content[:3])
                reconstructed += chunk.page_content[3:]
        self.assertEqual(reconstructed, text)
        self.assertEqual(document.metadata, {"source": "letters.txt"})

    def test_short_document(self):
        document = Document(page_content="Short note.", metadata={"source": "note.txt"})
        chunks = split_document(document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, "Short note.")

    def test_sample_document_pipeline(self):
        path = Path(__file__).resolve().parents[1] / "examples/documents/rag_intro.txt"
        document = load_text(path)
        chunks = split_document(document)
        self.assertGreater(len(chunks), 1)
        covered = [False] * len(document.page_content)
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = start + len(chunk.page_content)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(len(chunk.page_content), 500)
            self.assertEqual(document.page_content[start:end], chunk.page_content)
            self.assertEqual(chunk.metadata["source"], path.name)
            covered[start:end] = [True] * (end - start)
        self.assertTrue(
            all(seen or char.isspace() for char, seen in zip(document.page_content, covered))
        )

    def test_invalid_chunk_settings(self):
        document = Document(page_content="text")
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    split_document(document, chunk_size=size, chunk_overlap=overlap)


if __name__ == "__main__":
    unittest.main()
