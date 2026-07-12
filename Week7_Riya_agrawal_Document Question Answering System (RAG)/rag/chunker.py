"""
Text Chunking
=============
Splits raw document text into smaller overlapping chunks so retrieval can
pinpoint the most relevant passage instead of an entire document/page.
"""

import re
from dataclasses import dataclass, field
from typing import List

from .document_loader import Document


@dataclass
class Chunk:
    """A chunk of text ready to be embedded and stored."""
    text: str
    source: str
    page: int
    chunk_id: int
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> List[str]:
    """Very light-weight sentence splitter (no external NLP deps needed)."""
    # Split on sentence-ending punctuation followed by whitespace + capital/number,
    # while keeping the delimiter.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Split `text` into overlapping chunks of roughly `chunk_size` characters,
    trying to break on sentence boundaries so chunks stay semantically coherent.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk, carrying over the tail of the previous chunk
            # for context continuity (overlap)
            overlap_text = current[-chunk_overlap:] if current else ""
            current = f"{overlap_text} {sentence}".strip()

            # Edge case: a single sentence longer than chunk_size
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - chunk_overlap:]

    if current:
        chunks.append(current)

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """Chunk a list of Document objects into a flat list of Chunk objects."""
    all_chunks: List[Chunk] = []
    running_id = 0
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for piece in pieces:
            all_chunks.append(
                Chunk(
                    text=piece,
                    source=doc.source,
                    page=doc.page,
                    chunk_id=running_id,
                    metadata={"source": doc.source, "page": doc.page},
                )
            )
            running_id += 1
    return all_chunks
