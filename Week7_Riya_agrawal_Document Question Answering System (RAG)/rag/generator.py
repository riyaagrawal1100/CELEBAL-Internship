"""
Answer Generation
=================
Takes the user's question + retrieved context chunks and produces a final
grounded answer.

Three backends are supported (pick with `backend=`):

1. "extractive" (default, zero setup, no API key, no downloads)
   A lightweight, fully offline "generator" that selects and stitches
   together the most relevant sentences from the retrieved context. It
   won't write beautiful prose, but it is 100% grounded, free, and needs
   nothing beyond the standard library + scikit-learn.

2. "openai"
   Uses the OpenAI Chat Completions API. Requires `OPENAI_API_KEY` env var
   and the `openai` package.

3. "anthropic"
   Uses the Anthropic Messages API (Claude). Requires `ANTHROPIC_API_KEY`
   env var and the `anthropic` package.

The prompt template used for the LLM backends explicitly instructs the
model to answer ONLY from the provided context — this is the core idea of
RAG: grounding generation in retrieved evidence instead of parametric
memory.
"""

import os
import re
from typing import List

from .chunker import Chunk

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a specific "
    "document. Use ONLY the provided context to answer. If the answer is "
    "not contained in the context, say you don't have enough information "
    "in the document to answer. Be concise and cite which source/page the "
    "information came from when possible."
)


def _build_prompt(question: str, context_chunks: List[Chunk]) -> str:
    context_str = "\n\n".join(
        f"[Source: {c.source}, page {c.page}]\n{c.text}" for c in context_chunks
    )
    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above:"
    )


# ---------------------------------------------------------------------------
# Backend 1: Extractive (offline, no API key needed)
# ---------------------------------------------------------------------------
def _extractive_answer(question: str, context_chunks: List[Chunk]) -> str:
    """
    Rank sentences from the retrieved chunks by TF-IDF similarity to the
    question and stitch the top ones together into a short answer. This
    keeps the whole pipeline runnable with zero external API calls.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sentences = []
    sentence_sources = []
    for c in context_chunks:
        for s in re.split(r"(?<=[.!?])\s+", c.text):
            s = s.strip()
            if len(s) > 15:
                sentences.append(s)
                sentence_sources.append(f"{c.source} (p.{c.page})")

    if not sentences:
        return (
            "I couldn't find relevant information in the document to answer "
            "that question."
        )

    corpus = sentences + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()

    top_n = min(3, len(sentences))
    top_idx = sims.argsort()[::-1][:top_n]
    # keep original order for readability
    top_idx = sorted(top_idx)

    answer_sentences = [sentences[i] for i in top_idx]
    sources = sorted(set(sentence_sources[i] for i in top_idx))

    answer = " ".join(answer_sentences)
    answer += f"\n\n(Source: {', '.join(sources)})"
    return answer


# ---------------------------------------------------------------------------
# Backend 2: OpenAI
# ---------------------------------------------------------------------------
def _openai_answer(question: str, context_chunks: List[Chunk], model: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable to use this backend.")

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Backend 3: Anthropic (Claude)
# ---------------------------------------------------------------------------
def _anthropic_answer(question: str, context_chunks: List[Chunk], model: str) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to use this backend.")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(question, context_chunks)

    message = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def generate_answer(
    question: str,
    context_chunks: List[Chunk],
    backend: str = "extractive",
    model: str = None,
) -> str:
    """
    Generate a grounded answer to `question` using `context_chunks` as
    evidence.

    backend: "extractive" | "openai" | "anthropic"
    model:   overrides the default model name for the chosen backend
    """
    if not context_chunks:
        return "I couldn't find relevant information in the document to answer that question."

    if backend == "extractive":
        return _extractive_answer(question, context_chunks)
    elif backend == "openai":
        return _openai_answer(question, context_chunks, model or "gpt-4o-mini")
    elif backend == "anthropic":
        return _anthropic_answer(question, context_chunks, model or "claude-sonnet-4-6")
    else:
        raise ValueError(f"Unknown generation backend: {backend}")
