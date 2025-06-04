import streamlit as st
import os
from datetime import datetime
from config import Config
from document_processor import process_pdf, save_uploaded_file
from retriever import HybridRetriever
from llm_handler import LocalLLMHandler

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'document_content' not in st.session_state:
    st.session_state.document_content = {}
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

@st.cache_resource
def get_embeddings():
    if not EMBEDDINGS_AVAILABLE:
        st.error("Embeddings not available. Install sentence-transformers: pip install sentence-transformers")
        return None
    
    config = Config()
    try:
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None

@st.cache_resource
def get_llm_handler():
    return LocalLLMHandler()

st.set_page_config(page_title="PDF Q&A System", layout="wide")
st.title("📚 Local PDF Q&A System")
st.markdown("Upload PDFs and ask questions using local AI models (no API keys required)")

with st.sidebar:
    st.header("⚙️ System Status")
    
    if not st.session_state.get('models_loaded', False):
        with st.spinner("Loading AI models..."):
            try:
                st.session_state.llm_handler = get_llm_handler()
                st.session_state.models_loaded = True
                st.success("✅ AI models loaded")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                st.session_state.llm_handler = None
    else:
        st.success("✅ AI models ready")
    
    st.header("📤 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_path = save_uploaded_file(uploaded_file)
                    status, documents = process_pdf(file_path, uploaded_file.name)
                    
                    if status == "success":
                        st.session_state.uploaded_files[uploaded_file.name] = {
                            'path': file_path,
                            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'size': len(documents)
                        }
                        st.session_state.document_content[uploaded_file.name] = documents
                        st.success(f"✅ {uploaded_file.name} processed! ({len(documents)} chunks)")
                    else:
                        st.error(f"❌ Error processing {uploaded_file.name}: {status}")
    
    st.header("📄 Uploaded Files")
    if st.session_state.uploaded_files:
        for filename, info in st.session_state.uploaded_files.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {filename}")
                st.caption(f"Uploaded: {info['upload_time']} | Chunks: {info['size']}")
            with col2:
                if st.button("🗑️", key=f"delete_{filename}"):
                    os.remove(info['path'])
                    del st.session_state.uploaded_files[filename]
                    if filename in st.session_state.document_content:
                        del st.session_state.document_content[filename]
                    st.rerun()
    else:
        st.info("No files uploaded yet")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Ask Questions")
    
    if st.session_state.uploaded_files:
        selected_files = st.multiselect(
            "Select PDFs to query:",
            options=list(st.session_state.uploaded_files.keys()),
            default=list(st.session_state.uploaded_files.keys())
        )
        
        if selected_files and st.session_state.get('models_loaded', False) and st.session_state.get('llm_handler'):
            question = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic discussed in the documents?"
            )
            
            if question:
                with st.spinner("Searching and generating answer..."):
                    try:
                        all_documents = []
                        for filename in selected_files:
                            if filename in st.session_state.document_content:
                                all_documents.extend(st.session_state.document_content[filename])
                        
                        if all_documents:
                            embeddings = get_embeddings()
                            if embeddings:
                                retriever = HybridRetriever(all_documents, embeddings)
                                relevant_docs = retriever.retrieve(question)
                            else:
                                # Fallback to simple text search if embeddings fail
                                relevant_docs = []
                                query_words = set(question.lower().split())
                                for doc in all_documents:
                                    doc_words = set(doc.page_content.lower().split())
                                    if query_words.intersection(doc_words):
                                        relevant_docs.append(doc)
                                relevant_docs = relevant_docs[:3]  # Take top 3
                            
                            answer = st.session_state.llm_handler.generate_answer(question, relevant_docs)
                            
                            st.success("**Answer:**")
                            st.write(answer)
                            
                            with st.expander("📚 Source References"):
                                for i, doc in enumerate(relevant_docs, 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.markdown(f"- **File:** {doc.metadata.get('source', 'Unknown')}")
                                    st.markdown(f"- **Page:** {doc.metadata.get('page', 'Unknown')}")
                                    st.markdown(f"- **Type:** {doc.metadata.get('type', 'Unknown')}")
                                    st.markdown(f"- **Content:**")
                                    content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                                    st.text(content_preview)
                                    st.divider()
                        else:
                            st.warning("No documents found in selected files")
                    
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        else:
            st.info("📌 Select PDF files and wait for models to load")
    else:
        st.info("📤 Please upload PDF files using the sidebar")

with col2:
    st.header("📊 System Info")
    
    if st.session_state.uploaded_files:
        total_files = len(st.session_state.uploaded_files)
        total_chunks = sum(info['size'] for info in st.session_state.uploaded_files.values())
        
        st.metric("Total PDFs", total_files)
        st.metric("Total Chunks", total_chunks)
        st.metric("Avg Chunks/PDF", round(total_chunks / total_files, 1) if total_files > 0 else 0)
        
        st.info(
            "**System Features:**\n"
            "- Local AI (No API required)\n"
            "- Hybrid Search (Dense + BM25)\n"
            "- Embedding: MiniLM-L6\n"
            "- LLM: Local Transformer\n"
            "- Chunk Size: 500 chars"
        )
    else:
        st.info("Upload PDFs to see statistics")
    
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Specific Questions**: Ask clear, specific questions
        - **Multiple PDFs**: Select relevant documents only
        - **Table Queries**: Use keywords like "table", "data"
        - **Context**: Questions about specific pages work well
        - **Performance**: First query may be slower as models initialize
        """)

st.divider()
st.caption("Local PDF Q&A System | No external APIs required | Optimized for laptops")