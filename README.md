# AI Powered Financial Document QA Chatbot with RAG PDF Support
An **AI-powered financial chatbot** that answers questions from annual reports. Built using **LangChain**, **FAISS**, **Streamlit**, and **Mistral LLM via Ollama**, it supports both **preloaded company data** and **user-uploaded PDFs** and it works **completely offline** with **no API keys or cost**.

---

## 🚀 Features

- ✅ **Offline & Free** – No API keys, no cloud cost
- 🧠 **RAG Pipeline** – Retrieval Augmented Generation for better contextual responses
- 📁 **PDF Upload** – Users can upload their own annual reports
- 🔎 **Semantic Search** – Uses vector similarity with FAISS
- 📊 **Preloaded Reports** – Includes reports like Reliance, TCS
- 🖥️ **Streamlit Dashboard** – Simple UI for interaction

---

## 🛠️ Tech Stack

| Layer            | Tools Used                                      |
|------------------|-------------------------------------------------|
| LLM              | `Mistral` via `Ollama` (offline)                |
| Embeddings       | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)    |
| Vector DB        | `FAISS`                                         |
| Document Parsing | `PyPDF2`                                        |
| App UI           | `Streamlit`                                     |
| Framework        | `LangChain`                                     |

---

## 📸 Screenshots

<img width="1920" height="1080" alt="Screenshot 2025-06-20 005154" src="https://github.com/user-attachments/assets/4937478e-16d1-4059-a9af-6ab93cecd17d" />


---

## ⚙️ How to Run Locally

1. **Install dependencies**

```bash
pip install -r requirements.txt
Install Ollama and pull Mistral model

ollama run mistral
Start the Streamlit app

streamlit run financial_chatbot_ui.py
```

🧾 How It Works
Extracts text from uploaded or preloaded PDFs using PyPDF2

Splits the text and creates vector embeddings

Stores those embeddings in FAISS for semantic search

Uses Mistral (via Ollama) to generate answers based on retrieved chunks

Displays answers in a friendly Streamlit interface
