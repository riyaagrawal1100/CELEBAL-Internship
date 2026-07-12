"""
Streamlit UI for the RAG Document Question Answering System.

Run with:
    pip install streamlit
    streamlit run streamlit_app.py
"""

import os
import tempfile

import streamlit as st

from rag.pipeline import RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

st.title("📄 Document Question Answering System (RAG)")
st.caption(
    "Upload your own PDFs / text files and ask questions. Answers are "
    "grounded in retrieved passages from your documents."
)

with st.sidebar:
    st.header("⚙️ Settings")
    embedding_backend = st.selectbox(
        "Embedding backend", ["auto", "tfidf", "sentence-transformers"], index=0
    )
    generation_backend = st.selectbox(
        "Generation backend", ["extractive", "openai", "anthropic"], index=0
    )
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 4)
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 800, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 500, 150, step=50)
    st.markdown("---")
    st.markdown(
        "**Note:** `openai` / `anthropic` backends need the matching API key "
        "set as an environment variable (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`)."
    )

uploaded_files = st.file_uploader(
    "Upload PDF / TXT / MD files", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_files and st.button("📥 Ingest documents"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for uf in uploaded_files:
            path = os.path.join(tmp_dir, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            paths.append(path)

        with st.spinner("Loading, chunking, and embedding documents..."):
            pipeline = RAGPipeline(
                embedding_backend=embedding_backend,
                generation_backend=generation_backend,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
            )
            n_chunks = pipeline.ingest(paths)
            st.session_state.pipeline = pipeline

        st.success(
            f"Indexed {n_chunks} chunks from {len(paths)} file(s) "
            f"using '{pipeline.embedder.name}' embeddings."
        )

st.markdown("---")

if st.session_state.pipeline is None:
    st.info("👆 Upload one or more documents and click **Ingest documents** to get started.")
else:
    question = st.text_input("Ask a question about your document(s):")
    if st.button("🔍 Get answer") and question:
        with st.spinner("Retrieving context and generating answer..."):
            result = st.session_state.pipeline.ask(question)
        st.session_state.history.insert(0, result)

    for result in st.session_state.history:
        st.markdown(f"### ❓ {result['question']}")
        st.markdown(f"**Answer:** {result['answer']}")
        with st.expander("📚 Retrieved context / sources"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(
                    f"**[{i}] {src['source']} (page {src['page']}, "
                    f"similarity {src['score']})**"
                )
                st.write(src["text"])
        st.markdown("---")
