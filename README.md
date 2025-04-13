# 📚 Book QA using RAG with FAISS & OpenAI

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for question answering based on any book (in `.txt` format). If the user doesn't provide a file, it defaults to *Pride and Prejudice* by Jane Austen from Project Gutenberg.

---

## 🚀 Features

- 🔍 Ask any question about a book.
- 📂 Upload your own book as a `.txt` file OR use a default book (auto-downloaded).
- ✂️ Text chunking with overlap using LangChain.
- 🧠 Embedding generation using SentenceTransformers (`all-MiniLM-L6-v2`).
- ⚡ Fast semantic search with FAISS.
- 🤖 Answer generation using OpenAI GPT-4o-mini.
- 🌐 Intuitive Streamlit web interface.
- 🧠 FAISS index is saved with the book name for reuse (e.g., `pride_and_prejudice_index.faiss`).

---

## 🧱 Tech Stack

- Python
- Streamlit
- OpenAI GPT-4o-mini
- SentenceTransformers
- FAISS
- LangChain

---
