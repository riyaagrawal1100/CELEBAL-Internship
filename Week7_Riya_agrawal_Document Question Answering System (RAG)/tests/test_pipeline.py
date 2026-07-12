"""
Basic tests for the RAG pipeline. Run with:
    python -m pytest tests/ -v
or simply:
    python tests/test_pipeline.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.chunker import chunk_text, chunk_documents
from rag.document_loader import load_documents, Document
from rag.pipeline import RAGPipeline

SAMPLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_data",
    "sample_notes.txt",
)


class TestChunker(unittest.TestCase):
    def test_chunk_text_basic(self):
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertLessEqual(len(c), 50 + 20)  # small slack for overlap boundary

    def test_chunk_documents(self):
        docs = [Document(text="Sentence one. Sentence two. Sentence three.", source="test.txt")]
        chunks = chunk_documents(docs, chunk_size=20, chunk_overlap=5)
        self.assertTrue(all(c.source == "test.txt" for c in chunks))


class TestDocumentLoader(unittest.TestCase):
    def test_load_text_file(self):
        docs = load_documents([SAMPLE_PATH])
        self.assertEqual(len(docs), 1)
        self.assertIn("RAG", docs[0].text)


class TestRAGPipelineExtractive(unittest.TestCase):
    """End-to-end test using the zero-dependency TF-IDF + extractive backends."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RAGPipeline(
            embedding_backend="tfidf",
            generation_backend="extractive",
            chunk_size=500,
            chunk_overlap=100,
            top_k=3,
        )
        cls.n_chunks = cls.pipeline.ingest([SAMPLE_PATH])

    def test_ingest_produces_chunks(self):
        self.assertGreater(self.n_chunks, 0)

    def test_ask_returns_answer_and_sources(self):
        result = self.pipeline.ask("What are the main stages of a RAG pipeline?")
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertGreater(len(result["sources"]), 0)
        self.assertTrue(len(result["answer"]) > 0)

    def test_retrieval_relevance(self):
        # A question about embeddings should retrieve the embeddings section
        result = self.pipeline.ask("What is an embedding?", top_k=2)
        combined_context = " ".join(s["text"] for s in result["sources"]).lower()
        self.assertIn("embedding", combined_context)

    def test_unrelated_question_still_returns_something(self):
        result = self.pipeline.ask("What is the capital of France?")
        self.assertIn("answer", result)  # should not crash; extractive backend always returns text


if __name__ == "__main__":
    unittest.main(verbosity=2)
