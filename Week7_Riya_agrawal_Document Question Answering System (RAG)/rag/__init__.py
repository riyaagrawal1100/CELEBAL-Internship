"""
RAG (Retrieval-Augmented Generation) Document Question Answering System
=========================================================================

A lightweight, dependency-friendly RAG pipeline that lets you ask questions
over your own documents (PDF / TXT / MD).

Modules
-------
document_loader : Load & clean text from PDFs / text files
chunker         : Split raw text into overlapping chunks
embeddings      : Convert text chunks into vector representations
vector_store    : Store embeddings + do similarity search ("vector database")
generator       : Turn retrieved context + question into a final answer
pipeline        : Wires everything together end-to-end
"""

__version__ = "1.0.0"
