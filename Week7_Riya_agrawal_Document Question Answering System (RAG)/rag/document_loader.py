"""
Document Ingestion
===================
Loads documents (PDF, TXT, MD) from disk and returns cleaned raw text,
along with basic metadata (source filename, page number where applicable).
"""

import os
import re
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """A single loaded document unit (e.g. one page of a PDF, or one file)."""
    text: str
    source: str
    page: int = 0


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip weird control characters."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str) -> List[Document]:
    """Load a PDF file, returning one Document per page."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required to read PDF files. Install with: pip install pypdf"
        ) from e

    reader = PdfReader(path)
    docs = []
    source = os.path.basename(path)
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        cleaned = _clean_text(raw)
        if cleaned:
            docs.append(Document(text=cleaned, source=source, page=i + 1))
    return docs


def load_text_file(path: str) -> List[Document]:
    """Load a plain text / markdown file as a single Document."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    cleaned = _clean_text(raw)
    source = os.path.basename(path)
    return [Document(text=cleaned, source=source, page=0)] if cleaned else []


def load_document(path: str) -> List[Document]:
    """Dispatch loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".txt", ".md"):
        return load_text_file(path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md"
        )


def load_documents(paths: List[str]) -> List[Document]:
    """Load multiple documents/files and return a flat list of Document objects."""
    all_docs: List[Document] = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                fpath = os.path.join(p, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in (
                    ".pdf",
                    ".txt",
                    ".md",
                ):
                    all_docs.extend(load_document(fpath))
        else:
            all_docs.extend(load_document(p))
    return all_docs
