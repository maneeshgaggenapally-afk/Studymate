import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss  # FAISS for vector search
import re
import google.generativeai as genai
import time

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="StudyMate Pro - PDF Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Using a condensed CSS block for readability
st.markdown("""
<style>
    .main { background-color: #f0f2f6; color: #2c3e50; }
    .sidebar .sidebar-content { background: #34495e; color: white; }
    .stButton>button { background: #3498db; color: white; border-radius: 12px; padding: 12px 28px; border: none; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton>button:hover { background: #2980b9; transform: translateY(-2px); box-shadow: 0 6px 10px rgba(0,0,0,0.2); }
    .answer-box { background: white; padding: 25px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); margin-bottom: 25px; border-left: 6px solid #3498db; line-height: 1.6; }
    .source-box { background: #ecf0f1; padding: 18px; border-radius: 12px; border-left: 4px solid #2c3e50; margin-bottom: 15px; }
    .header { color: #2c3e50; text-align: center; margin-bottom: 10px; font-weight: 700; }
    .subheader { color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 8px; font-weight: 600; }
    .info-box { background: #e3f2fd; color: #1565c0; padding: 18px; border-radius: 12px; margin-bottom: 20px; border-left: 5px solid #1976d2; }
    .success-box { background: #e8f5e9; color: #2e7d32; padding: 18px; border-radius: 12px; margin-bottom: 25px; border-left: 5px solid #43a047; }
</style>
""", unsafe_allow_html=True)


# --- Session State Management ---
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks_with_metadata' not in st.session_state:
    st.session_state.chunks_with_metadata = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'file_names' not in st.session_state:
    st.session_state.file_names = []

# --- API Key Configuration ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    GEMINI_API_CONFIGURED = True
except (KeyError, Exception):
    GEMINI_API_CONFIGURED = False

# --- Helper Functions ---

def extract_text_from_pdfs(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page_num, page in enumerate(doc):
                documents.append({
                    "text": page.get_text("text"),
                    "filename": uploaded_file.name,
                    "page_num": page_num + 1
                })
            doc.close()
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
    return documents

def preprocess_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    return text

def chunk_documents(documents: list, chunk_size: int = 400, overlap: int = 80):
    chunks_with_metadata = []
    for doc in documents:
        text = preprocess_text(doc["text"])
        if not text: continue
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            chunks_with_metadata.append({
                "text": chunk_text,
                "source": f"{doc['filename']}, Page {doc['page_num']}"
            })
    return chunks_with_metadata

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_faiss_index(chunks_with_metadata: list):
    model = load_embedding_model()
    text_chunks = [chunk['text'] for chunk in chunks_with_metadata]
    embeddings = model.encode(text_chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def semantic_search_faiss(question: str, index: faiss.Index, chunks_with_metadata: list, top_k: int = 5):
    model = load_embedding_model()
    question_embedding = model.encode([question])
    _, I = index.search(question_embedding, top_k)
    return [chunks_with_metadata[i] for i in I[0] if i != -1]

def generate_answer_with_gemini(question: str, relevant_chunks: list):
    if not GEMINI_API_CONFIGURED:
        st.error("🚨 Gemini API Key is not configured. Please add it to your .streamlit/secrets.toml file.")
        return "API key not configured."
    if not relevant_chunks:
        return "I couldn't find relevant information in your documents to answer that question."

    context = "\n\n---\n\n".join([f"Source: {chunk['source']}\nContent: {chunk['text']}" for chunk in relevant_chunks])
    prompt = f"Answer the following question based ONLY on the provided context. Cite your sources (e.g., [Source: file.pdf, Page 5]). If the answer is not in the context, say so.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    
    try:
        # --- THIS IS THE CORRECTED LINE ---
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return "There was an issue generating the answer."

# --- Streamlit App Layout ---
st.markdown("<h1 class='header'>🧠 StudyMate Pro: Advanced PDF Q&A</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='color: white;'>📁 Upload Documents</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Select PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    chunk_size = st.slider("Context Chunk Size (words)", 200, 800, 400)
    top_k = st.slider("Number of Source Passages", 1, 10, 4)
    
    st.markdown("---")
    process_button = st.button("🚀 Process Documents", use_container_width=True, disabled=not uploaded_files)
    
    if st.button("🔄 Clear All", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    if GEMINI_API_CONFIGURED:
        st.success("✅ Gemini API Key Loaded")
    else:
        st.error("❌ Gemini API Key Not Found")
        st.info("Please create a `.streamlit/secrets.toml` file with your GEMINI_API_KEY.")

# --- Main Content Area ---
if process_button:
    with st.spinner("Processing documents... This may take a moment."):
        documents = extract_text_from_pdfs(uploaded_files)
        if documents:
            st.session_state.chunks_with_metadata = chunk_documents(documents, chunk_size=chunk_size)
            st.session_state.faiss_index = create_faiss_index(st.session_state.chunks_with_metadata)
            st.session_state.file_names = [f.name for f in uploaded_files]
            st.session_state.processed = True
        else:
            st.error("Could not extract any text from the uploaded PDFs.")

if not st.session_state.processed:
    st.markdown("""
    <div class="info-box">
        <h4>📚 How to Use StudyMate Pro</h4>
        <ol>
            <li>Make sure your Gemini API key is in <code>.streamlit/secrets.toml</code>.</li>
            <li>Upload one or more PDF study materials in the sidebar.</li>
            <li>Click "Process Documents" to build the knowledge base.</li>
            <li>Ask a question to get a synthesized answer with source citations.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<h3 class='subheader'>Ask a Question</h3>", unsafe_allow_html=True)
    question = st.text_input("Enter your question:", placeholder="e.g., What are the key stages of photosynthesis?", label_visibility="collapsed")
    
    if st.button("💡 Get Answer", use_container_width=True, disabled=not question):
        with st.spinner("Searching and generating answer..."):
            relevant_chunks = semantic_search_faiss(question, st.session_state.faiss_index, st.session_state.chunks_with_metadata, top_k=top_k)
            answer = generate_answer_with_gemini(question, relevant_chunks)
            
            st.markdown("<h3 class='subheader'>Answer</h3>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            
            if relevant_chunks:
                st.markdown("<h3 class='subheader'>Source Passages</h3>", unsafe_allow_html=True)
                for i, chunk in enumerate(relevant_chunks):
                    st.markdown(f"""
                    <div class="source-box">
                        <h4>📝 Passage {i+1} (Source: {chunk['source']})</h4>
                        <p>{chunk['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #7f8c8d;'>Powered by Google Gemini</p>", unsafe_allow_html=True)

