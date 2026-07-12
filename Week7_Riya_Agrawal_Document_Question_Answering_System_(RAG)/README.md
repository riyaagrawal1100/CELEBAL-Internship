# 📄 Document Question Answering System using RAG

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) based Document Question Answering System. It allows users to upload a PDF or text document and ask questions related to its content. The system retrieves the most relevant information from the uploaded document and generates answers based on the retrieved context.

The objective of this project is to understand the working of RAG, embeddings, vector databases, and language models for document-based question answering.

---

## Features

- Upload PDF or text documents
- Ask questions related to uploaded documents
- Retrieve relevant information using semantic search
- Generate context-aware answers
- Simple and interactive Streamlit interface

---

## Technologies Used

- Python
- Streamlit
- LangChain
- FAISS
- Sentence Transformers
- PyPDF
- NumPy

---

## Project Structure

```text
Week7_Riya_agrawal_Document Question Answering System (RAG)/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
├── src/
└── assets/
```

---

## How It Works

1. Upload a PDF or text document.
2. Extract text from the document.
3. Split the text into smaller chunks.
4. Convert each chunk into embeddings.
5. Store the embeddings in a vector database.
6. Enter a question related to the document.
7. Retrieve the most relevant text chunks.
8. Generate the final answer using the retrieved context.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CELEBAL-Internship.git
```

Go to the project directory:

```bash
cd "Week7_Riya_agrawal_Document Question Answering System (RAG)"
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Sample Questions

- What is the main topic of the document?
- Summarize the document.
- What are the important points discussed?
- Explain a specific concept mentioned in the document.

---

## Future Improvements

- Support multiple PDF uploads
- Add chat history
- Improve the user interface
- Add source citations
- Support DOCX and PowerPoint files

---

## Learning Outcomes

Through this project, I learned:

- Basics of Retrieval-Augmented Generation (RAG)
- Working with embeddings and semantic search
- Using FAISS for vector similarity search
- Building a simple document question answering system
- Creating an interactive web application using Streamlit

---

## Author

**Riya Agrawal**  

---
