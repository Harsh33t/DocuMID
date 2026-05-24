import streamlit as st
import fitz  # PyMuPDF
import tiktoken
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Set page config
st.set_page_config(
    page_title="DocuMind - Chat with your PDF",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Custom CSS for modern glassmorphism dark styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Apply modern font and background styling */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Custom Gradient Title */
.title-container {
    padding: 1rem 0;
    margin-bottom: 2rem;
}
.title-main {
    background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin: 0;
    letter-spacing: -0.03em;
}
.title-sub {
    color: #94a3b8;
    font-size: 1.1rem;
    margin-top: 0.25rem;
    font-weight: 300;
}

/* Glassmorphism card utility */
.glass-panel {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Custom badges */
.badge-success {
    background-color: rgba(16, 185, 129, 0.1);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-block;
}
.badge-warning {
    background-color: rgba(245, 158, 11, 0.1);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.2);
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
    display: inline-block;
}

/* Streamlit elements cleanup */
[data-testid="stSidebar"] .element-container {
    margin-bottom: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Helper function for PDF parsing
def parse_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

# Helper function for chunking
def chunk_text(text, chunk_size=500, overlap=50):
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks

# Helper to create embeddings and FAISS index
def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

# Helper to retrieve top chunks
def retrieve_chunks(query, model, index, chunks, k=3):
    query_emb = model.encode([query])
    query_emb = np.array(query_emb).astype('float32')
    faiss.normalize_L2(query_emb)
    
    distances, indices = index.search(query_emb, k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx < len(chunks) and idx >= 0:
            results.append({
                "text": chunks[idx],
                "score": float(score)
            })
    return results

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# App Layout
st.markdown('<div class="title-container"><h1 class="title-main">🧠 DocuMind</h1><div class="title-sub">Intelligent retrieval-augmented Q&A for your PDFs</div></div>', unsafe_allow_html=True)

# Loading Embeddings Model
with st.spinner("Initializing neural network..."):
    embed_model = load_embedding_model()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Connection & API Status")
    if groq_api_key:
        st.markdown('<span class="badge-success">Groq API Key Active ✅</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warning">Missing Groq API Key ⚠️</span>', unsafe_allow_html=True)
        st.warning("Please specify your GROQ_API_KEY in the `.env` file to chat with PDFs.")
        
    st.markdown("---")
    st.markdown("### 📄 Document Upload")
    
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Extracting text and indexing..."):
                try:
                    file_bytes = uploaded_file.read()
                    pdf_text = parse_pdf(file_bytes)
                    
                    if not pdf_text.strip():
                        st.error("No extractable text found in PDF.")
                    else:
                        chunks = chunk_text(pdf_text)
                        index = build_index(chunks, embed_model)
                        
                        st.session_state.processed_file = uploaded_file.name
                        st.session_state.chunks = chunks
                        st.session_state.faiss_index = index
                        st.session_state.chunk_count = len(chunks)
                        st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    
    # Document Info Panel
    if st.session_state.processed_file:
        st.markdown("### 📊 Document Metadata")
        st.markdown(f"**Filename:** `{st.session_state.processed_file}`")
        st.markdown(f"**Total Chunks:** `{st.session_state.chunk_count}`")
        
        # Clear Chat Button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Main Chat View
if not st.session_state.processed_file:
    # Beautiful welcome screen when no document is uploaded
    st.markdown(
        """
        <div class="glass-panel" style="margin-top: 2rem;">
            <h3>👋 Welcome to DocuMind!</h3>
            <p>DocuMind lets you query long PDF documents using cutting-edge Retrieval-Augmented Generation (RAG).</p>
            <hr style="border-top: 1px solid rgba(255,255,255,0.1);">
            <h5>How to get started:</h5>
            <ol>
                <li>Ensure you have configured your <code>GROQ_API_KEY</code> in the <code>.env</code> file.</li>
                <li>Upload a PDF document using the sidebar file uploader.</li>
                <li>Wait a few seconds for the document to be parsed, chunked, and embedded into our local FAISS index.</li>
                <li>Start asking questions directly in the chat input below!</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, source in enumerate(msg["sources"]):
                        st.markdown(f"**Source {idx+1}** (Similarity: `{source['score']:.4f}`)")
                        st.info(source["text"])
                        
    # Chat Input
    if prompt := st.chat_input("Ask a question about the document..."):
        if not groq_api_key:
            st.error("GROQ_API_KEY is missing. Please add it to your `.env` file.")
        else:
            # Render user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Retrieve relevant sources
            with st.spinner("Searching document..."):
                sources = retrieve_chunks(
                    prompt, 
                    embed_model, 
                    st.session_state.faiss_index, 
                    st.session_state.chunks
                )
                
            # Call Groq API and stream response
            try:
                client = Groq(api_key=groq_api_key)
                
                # Assemble system prompt context
                context_str = "\n\n---\n\n".join([s["text"] for s in sources])
                system_prompt = (
                    "You are DocuMind, an expert document assistant. Answer the user's question based "
                    "only on the provided PDF context below. If the answer cannot be found in the context, "
                    "state that you don't know based on the document. Keep your answers concise, "
                    "accurate, and helpful.\n\n"
                    f"### PDF Context:\n{context_str}"
                )
                
                # Compile complete messages list
                api_messages = [{"role": "system", "content": system_prompt}]
                
                # Add historical context (exclude system, limit history length if desired)
                for msg in st.session_state.messages[:-1]:  # exclude the user's latest prompt
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                # Add current user prompt
                api_messages.append({"role": "user", "content": prompt})
                
                # Send stream request to Groq
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=api_messages,
                    stream=True,
                )
                
                # Stream the response
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    
                    # Display retrieved sources in an expander
                    with st.expander("🔍 View Retrieved Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"**Source {idx+1}** (Similarity: `{source['score']:.4f}`)")
                            st.info(source["text"])
                            
                # Save assistant message and its sources to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                
            except Exception as e:
                st.error(f"Error communicating with Groq API: {str(e)}")
