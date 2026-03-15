import streamlit as st
import warnings
import os
import re
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from groq import Groq

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(page_title="Pro RAG Assistant", layout="wide", page_icon="🤖")
st.title("🤖 Advanced Dynamic RAG Assistant")
st.markdown("---")

# -----------------------------
# 2. RAG Logic (Cached)
# -----------------------------
@st.cache_resource
def process_docs(uploaded_files):
    """Processes uploaded PDFs and creates Hybrid Search Index."""
    all_docs = []
    
    for uploaded_file in uploaded_files:
        # Create a temporary file to allow PyPDFLoader to read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        all_docs.extend(loader.load())
        os.unlink(tmp_path) # Delete the temp file

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    chunks = splitter.split_documents(all_docs)
    corpus = [c.page_content for c in chunks]

    # Vector Search (Semantic)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # BM25 (Keyword Search)
    tokenized_corpus = [doc.lower().split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Re-ranker (Cross-Encoder)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return retriever, corpus, bm25, reranker

# -----------------------------
# 3. Sidebar & File Handling
# -----------------------------
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Check for API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file!")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Global Variables
retriever, corpus, bm25, reranker = None, None, None, None

if uploaded_files:
    with st.status("🛠️ Indexing Knowledge Base...", expanded=False) as status:
        retriever, corpus, bm25, reranker = process_docs(uploaded_files)
        status.update(label="✅ Indexing Complete!", state="complete")
else:
    st.info("👋 Welcome! Please upload one or more PDFs to begin.")
    st.stop()

# -----------------------------
# 4. Chat Memory & UI
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# 5. Advanced Query Expansion Logic
# -----------------------------
def get_expanded_queries(query):
    """Generates 3 variations of the query to catch more context."""
    prompt = f"Generate 3 short search queries for: {query}. Respond with ONLY the queries, one per line."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    lines = response.choices[0].message.content.split("\n")
    return [re.sub(r"^\d+\.\s*", "", q).strip() for q in lines if q.strip()]

# -----------------------------
# 6. Chat Input & Processing
# -----------------------------
if query := st.chat_input("Ask about your documents..."):
    # 1. Display User Message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🤖 Thinking..."):
        try:
            # A. Hybrid Retrieval
            exp_queries = [query] + get_expanded_queries(query)
            
            # Vector Search results
            vector_results = []
            for q in exp_queries:
                vector_results.extend([d.page_content for d in retriever.invoke(q)])
            
            # Keyword Search results (BM25)
            tokenized_query = query.lower().split(" ")
            bm25_indices = sorted(range(len(corpus)), key=lambda i: bm25.get_scores(tokenized_query)[i], reverse=True)[:5]
            keyword_results = [corpus[i] for i in bm25_indices]

            # B. Re-ranking
            combined = list(set(vector_results + keyword_results))
            pairs = [[query, doc] for doc in combined]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(scores, combined), key=lambda x: x[0], reverse=True)
            
            # Select top 3 best chunks
            top_context = "\n\n".join([text for score, text in ranked[:3]])

            # C. Generation
            system_instr = "You answer using ONLY the provided context. Be concise and accurate."
            history = st.session_state.messages[-4:] # Last 4 messages for memory
            
            final_messages = [{"role": "system", "content": system_instr}]
            final_messages.extend(history)
            final_messages.append({"role": "user", "content": f"Context:\n{top_context}\n\nQuestion: {query}"})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=final_messages,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content

            # D. Display & Save
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("🔍 View Sources"):
                    st.write(top_context)

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Something went wrong: {e}")