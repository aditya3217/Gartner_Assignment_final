# 📘 Book QA using RAG + Knowledge Graph (Neo4j)

This project implements a **hybrid QA system** that combines **Retrieval-Augmented Generation (RAG)** with a **Knowledge Graph (KG)** to answer questions about a book. If a book is not uploaded by the user, the system defaults to *Pride and Prejudice* by Jane Austen (sourced from Project Gutenberg).

---

## 🚀 Features

### 🔍 Question Answering System

- Ask any question about a book.
- Choose a book from dropdown:  
  - 📖 Default: *Pride and Prejudice*  
  - 📤 Upload your own `.txt` file
- Text chunking with overlap using **LangChain**
- Embedding generation via **SentenceTransformers (all-MiniLM-L6-v2)**
- Semantic search powered by **FAISS**
- Answer generation using **OpenAI GPT-4o-mini**

### 🧠 Knowledge Graph Enhancement

- Extracts semantic triples (subject-predicate-object) using **LLM-based extraction**
- Triples stored in **Neo4j** graph database as nodes and relationships
- Internally enriches the answer generation process by retrieving contextual information from the knowledge graph
- No manual Cypher query needed — KG context is automatically used to improve QA results

---

## 🧱 Tech Stack

- Python
- Streamlit
- FAISS
- OpenAI GPT-4o-mini
- SentenceTransformers
- LangChain
- Neo4j (via `py2neo`)
