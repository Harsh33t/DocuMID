# 🧠 DocuMind

An elegant, real-time PDF Q&A assistant built with Streamlit, sentence-transformers, FAISS, and Groq's high-speed API.

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Groq API](https://img.shields.io/badge/Groq-F59E0B?style=for-the-badge&logo=probot&logoColor=white)](https://groq.com)
[![FAISS VectorDB](https://img.shields.io/badge/FAISS-3B82F6?style=for-the-badge&logo=databricks&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-8B5CF6?style=for-the-badge&logo=huggingface&logoColor=white)](https://sbert.net)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-10B981?style=for-the-badge&logo=pdf&logoColor=white)](https://pymupdf.readthedocs.io/)

---

## 📽️ Demo

![DocuMind Demo Animation](https://raw.githubusercontent.com/Harsh33t/PortAI/main/documind/assets/demo.gif)



https://github.com/user-attachments/assets/95e02a7e-888a-41bc-a9bf-d1568ad7b023


---

## 📝 What It Does

DocuMind lets you upload any PDF document, automatically chunks and indexes its content in a local high-performance vector database, and lets you ask complex questions about the document in an interactive chat. It retrieves exact relevant contexts using semantic search and streams the answer using the high-performance Llama 3 model on Groq.

---

## 🧬 How It Works

DocuMind uses a modern **Retrieval-Augmented Generation (RAG)** pipeline to query your documents securely and fast:

```mermaid
graph TD
    A[Upload PDF] --> B[Parse PDF text using PyMuPDF]
    B --> C[Token-based chunking with overlap - 500 tokens]
    C --> D[Generate embeddings using all-MiniLM-L6-v2]
    D --> E[Build Local FAISS Vector Index]
    F[User Questions] --> G[Query Embedding]
    G --> H[Retrieve Top-3 Cosine Similarity Chunks from FAISS]
    H --> I[Format Prompt with Context + Query]
    I --> J[Groq API llama-3.1-8b-instant]
    J --> K[Stream Answer to Chat UI]
```

---

## ⚡ Setup in 3 Steps

Follow these simple steps to run DocuMind locally on your system:

### Step 1: Clone and Navigate
```bash
git clone https://github.com/Harsh33t/PortAI.git
cd stock_analysis/documind
```

### Step 2: Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### Step 3: Configure and Run
Create a `.env` file in the `documind` folder (or edit the template `.env`) and add your Groq API Key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_goes_here
```
Now, launch the Streamlit server:
```bash
streamlit run app.py
```

---

## 🚀 Future Improvements

- **Multi-PDF Indexing**: Support uploading and indexing multiple PDFs at once to query across your entire digital library.
- **User Authentication**: Implement OAuth/Supabase Auth so users can save their documents and chat histories securely.
- **Cloud Vector DB**: Migrate from local FAISS memory to a cloud vector database like Supabase Vector or Pinecone for persistence.
