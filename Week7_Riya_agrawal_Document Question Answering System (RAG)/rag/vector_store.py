"""
Vector Database
===============
A minimal, dependency-free "vector database": stores chunk embeddings in a
numpy array and performs cosine-similarity search to retrieve the most
relevant chunks for a query embedding.

For larger-scale / production use you could swap this out for FAISS,
Chroma, Pinecone, Weaviate, etc. — the `VectorStore` interface
(`add`, `search`, `save`, `load`) is designed to be a drop-in replacement.
"""

import json
import os
import pickle
from dataclasses import asdict
from typing import List, Tuple

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray = None  # shape: (n_chunks, dim)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their corresponding embedding vectors to the store."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks must match number of vectors")

        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """
        Return the top_k (chunk, similarity_score) pairs most similar to
        the query vector, using cosine similarity.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vector = query_vector.reshape(1, -1)
        # vectors are assumed L2-normalized already (see embeddings.py),
        # so dot product == cosine similarity.
        scores = (self.vectors @ query_vector.T).flatten()

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self.chunks)

    def save(self, path: str) -> None:
        """Persist the vector store to disk (numpy vectors + chunk metadata)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectors": self.vectors,
                    "chunks": [asdict(c) for c in self.chunks],
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.vectors = data["vectors"]
        store.chunks = [Chunk(**c) for c in data["chunks"]]
        return store
