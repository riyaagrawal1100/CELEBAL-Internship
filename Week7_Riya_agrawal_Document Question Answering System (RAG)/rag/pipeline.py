"""
RAG Pipeline
============
Wires together document loading -> chunking -> embedding -> vector store
-> retrieval -> generation into one easy-to-use class.

Example
-------
>>> from rag.pipeline import RAGPipeline
>>> pipeline = RAGPipeline()
>>> pipeline.ingest(["sample_data/sample_notes.txt"])
>>> result = pipeline.ask("What is the main idea of the document?")
>>> print(result["answer"])
"""

from typing import List, Dict, Any

from .document_loader import load_documents
from .chunker import chunk_documents
from .embeddings import get_embedder
from .vector_store import VectorStore
from .generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        embedding_backend: str = "auto",
        generation_backend: str = "extractive",
        generation_model: str = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 4,
    ):
        self.embedder = get_embedder(embedding_backend)
        self.generation_backend = generation_backend
        self.generation_model = generation_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.store = VectorStore()
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, paths: List[str]) -> int:
        """
        Load documents from `paths` (files or directories), chunk them,
        embed the chunks, and add them to the vector store.

        Returns the number of chunks added.
        """
        documents = load_documents(paths)
        if not documents:
            raise ValueError(f"No readable documents found in: {paths}")

        chunks = chunk_documents(
            documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        if not chunks:
            raise ValueError("Documents were loaded but produced no text chunks.")

        texts = [c.text for c in chunks]

        # TF-IDF backend needs to be fit on the corpus; neural embedders
        # ignore fit() (see embeddings.py).
        if not self._is_fitted:
            self.embedder.fit(texts)
            self._is_fitted = True

        vectors = self.embedder.encode(texts)
        self.store.add(chunks, vectors)
        return len(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def ask(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Ask a question. Returns a dict with the answer, the retrieved
        context chunks, and their similarity scores.
        """
        if len(self.store) == 0:
            raise RuntimeError("No documents have been ingested yet. Call ingest() first.")

        k = top_k or self.top_k
        query_vector = self.embedder.encode([question])[0]
        results = self.store.search(query_vector, top_k=k)

        context_chunks = [chunk for chunk, score in results]
        answer = generate_answer(
            question,
            context_chunks,
            backend=self.generation_backend,
            model=self.generation_model,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": round(score, 4),
                    "text": chunk.text,
                }
                for chunk, score in results
            ],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        self.store.save(path)

    def load(self, path: str) -> None:
        self.store = VectorStore.load(path)
        self._is_fitted = True
