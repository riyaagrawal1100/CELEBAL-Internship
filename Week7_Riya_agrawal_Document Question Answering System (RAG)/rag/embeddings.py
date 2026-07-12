"""
Embedding Creation
==================
Converts text chunks into vector representations capturing semantic meaning.

Two backends are supported:

1. "tfidf"  (default, zero setup)
   Uses scikit-learn's TF-IDF vectorizer + SVD for a lightweight, fully
   offline embedding that needs no downloads and no API keys. Great for
   getting the whole pipeline running immediately.

2. "sentence-transformers" (optional, better quality)
   Uses a real neural embedding model (e.g. "all-MiniLM-L6-v2") if the
   `sentence-transformers` package is installed. Falls back to TF-IDF
   automatically if it isn't available.

Both backends implement the same interface: `fit(texts)` and
`encode(texts) -> np.ndarray`, so the rest of the pipeline never needs to
know which one is active.
"""

from typing import List
import numpy as np


class TfidfEmbedder:
    """Offline, dependency-light embedding backend using TF-IDF + SVD."""

    name = "tfidf"

    def __init__(self, n_components: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )
        self.n_components = n_components
        self.svd = None  # created in fit(), sized to the corpus
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        tfidf_matrix = self.vectorizer.fit_transform(texts)

        # TruncatedSVD needs at least 2 samples and n_components < n_features.
        # For tiny corpora (e.g. a single-chunk document), skip SVD entirely
        # and use the raw (dense) TF-IDF vectors instead.
        max_components = min(tfidf_matrix.shape) - 1
        if tfidf_matrix.shape[0] < 2 or max_components < 1:
            self.svd = None
            self._fitted = True
            return

        n_components = min(self.n_components, max_components)
        from sklearn.decomposition import TruncatedSVD
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(tfidf_matrix)
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(texts) before encode(texts).")
        tfidf_matrix = self.vectorizer.transform(texts)

        if self.svd is not None:
            vectors = self.svd.transform(tfidf_matrix)
        else:
            vectors = tfidf_matrix.toarray()

        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return vectors / norms


class SentenceTransformerEmbedder:
    """High-quality neural embedding backend (requires sentence-transformers)."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: F401

        self.model = SentenceTransformer(model_name)
        self._fitted = True  # no fitting needed for pretrained models

    def fit(self, texts: List[str]) -> None:
        # Nothing to fit; pretrained model is used directly.
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors)


def get_embedder(backend: str = "auto", **kwargs):
    """
    Factory for embedding backends.

    backend:
      - "auto": try sentence-transformers, fall back to tfidf
      - "sentence-transformers": force neural embeddings (raises if not installed)
      - "tfidf": force lightweight offline embeddings
    """
    if backend == "tfidf":
        return TfidfEmbedder(**kwargs)

    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(**kwargs)

    if backend == "auto":
        try:
            return SentenceTransformerEmbedder(**kwargs)
        except ImportError:
            print(
                "[embeddings] sentence-transformers not installed — "
                "falling back to the built-in TF-IDF embedder. "
                "(pip install sentence-transformers for better quality)"
            )
            return TfidfEmbedder()

    raise ValueError(f"Unknown embedding backend: {backend}")
