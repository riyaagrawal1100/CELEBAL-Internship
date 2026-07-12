#!/usr/bin/env python3
"""
CLI for the RAG Document Question Answering System.

Usage
-----
# One-off question
python app.py --docs sample_data/sample_notes.txt --question "What is this document about?"

# Interactive chat mode over one or more documents / a folder
python app.py --docs sample_data/ --interactive

# Use a real neural embedding model + Claude for generation (requires
# `pip install sentence-transformers anthropic` and ANTHROPIC_API_KEY set)
python app.py --docs mynotes.pdf --interactive \
    --embedding-backend sentence-transformers \
    --generation-backend anthropic
"""

import argparse
import sys

from rag.pipeline import RAGPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions over your own documents using RAG."
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Path(s) to PDF/TXT/MD files, or a directory containing them.",
    )
    parser.add_argument("--question", "-q", type=str, default=None, help="A single question to ask.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive Q&A loop.")
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "tfidf", "sentence-transformers"],
        default="auto",
        help="Embedding backend. 'auto' tries sentence-transformers, falls back to tfidf.",
    )
    parser.add_argument(
        "--generation-backend",
        choices=["extractive", "openai", "anthropic"],
        default="extractive",
        help="How to generate the final answer from retrieved context.",
    )
    parser.add_argument("--generation-model", type=str, default=None, help="Override model name for openai/anthropic backend.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per question.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters.")
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved source chunks alongside the answer.")
    return parser


def print_result(result: dict, show_sources: bool) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {result['question']}")
    print("-" * 70)
    print(f"A: {result['answer']}")
    if show_sources:
        print("-" * 70)
        print("Retrieved context:")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source']} (page {src['page']}, score {src['score']})")
            snippet = src["text"][:200].replace("\n", " ")
            print(f"      \"{snippet}...\"")
    print("=" * 70)


def main() -> None:
    args = build_arg_parser().parse_args()

    print(f"[1/2] Loading & indexing documents: {args.docs}")
    pipeline = RAGPipeline(
        embedding_backend=args.embedding_backend,
        generation_backend=args.generation_backend,
        generation_model=args.generation_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    try:
        n_chunks = pipeline.ingest(args.docs)
    except Exception as e:
        print(f"Error ingesting documents: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[2/2] Indexed {n_chunks} chunks using '{pipeline.embedder.name}' embeddings.")
    print(f"Generation backend: {args.generation_backend}")

    if args.question:
        result = pipeline.ask(args.question)
        print_result(result, args.show_sources)
        return

    if args.interactive or not args.question:
        print("\nInteractive mode. Type your question and press Enter.")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                question = input("Your question> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            result = pipeline.ask(question)
            print_result(result, args.show_sources)


if __name__ == "__main__":
    main()
