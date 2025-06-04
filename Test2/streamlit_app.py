import streamlit as st
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from qa_system import QASystem
from config import Config

st.set_page_config(
    page_title="Local PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_qa_system():
    return QASystem()

def save_uploaded_file(uploaded_file, upload_dir: Path) -> str:
    file_path = upload_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

def display_document_content(content_list: List[Dict[str, Any]]):
    for item in content_list:
        content_type = item['type']
        
        if content_type == 'text':
            st.text_area(
                f"Text Content",
                value=item['content'],
                height=200,
                disabled=True
            )
        elif content_type == 'table':
            st.markdown("**Table Content:**")
            st.text(item['content'])
        elif content_type == 'image':
            st.markdown("**Image Content (OCR):**")
            st.text(item['content'])
            if 'image_path' in item['metadata']:
                try:
                    st.image(item['metadata']['image_path'], caption=f"Image from page")
                except:
                    st.warning("Could not display image")

def main():
    st.title("📚 Local PDF Q&A System")
    st.markdown("Upload PDFs and ask questions using local AI models - no API keys required!")
    
    if 'qa_system' not in st.session_state:
        with st.spinner("Initializing AI models..."):
            st.session_state.qa_system = initialize_qa_system()
    
    qa_system = st.session_state.qa_system
    
    with st.sidebar:
        st.header("📤 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple PDF files to create a knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_key = f"processed_{uploaded_file.name}"
                
                if file_key not in st.session_state:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        file_path = save_uploaded_file(uploaded_file, Config.UPLOAD_DIR)
                        result = qa_system.process_pdf(file_path, uploaded_file.name)
                        
                        st.session_state[file_key] = result
                        
                        if result["status"] == "success":
                            st.success(f"✅ {uploaded_file.name} processed!")
                            st.write(f"Chunks: {result['chunks']}")
                            st.write(f"Content types: {result['content_types']}")
                        else:
                            st.error(f"❌ Error: {result['message']}")
        
        st.divider()
        
        st.header("📊 Document Statistics")
        stats = qa_system.get_document_stats()
        
        if stats["total_documents"] > 0:
            st.metric("Total Documents", stats["total_documents"])
            st.metric("Total Chunks", stats["total_chunks"])
            
            st.subheader("Content Types")
            for content_type, count in stats["content_types"].items():
                st.write(f"• {content_type.title()}: {count}")
            
            st.subheader("Loaded Documents")
            for doc in stats["documents"]:
                with st.expander(f"📄 {doc['filename']}"):
                    st.write(f"**Pages:** {doc['total_pages']}")
                    st.write(f"**Chunks:** {doc['total_chunks']}")
                    st.write(f"**Processed:** {doc['processed_at']}")
                    
                    if st.button(f"Remove {doc['filename']}", key=f"remove_{doc['filename']}"):
                        qa_system.remove_document(doc['filename'])
                        st.rerun()
        else:
            st.info("No documents loaded yet")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Ask Questions", "📖 Browse Documents", "🔍 Search", "📊 Analytics"])
    
    with tab1:
        st.header("Ask Questions About Your Documents")
        
        stats = qa_system.get_document_stats()
        if stats["total_documents"] == 0:
            st.warning("Please upload PDF documents first!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="What is the main topic discussed in the documents?",
                help="Ask specific questions for better results"
            )
        
        with col2:
            selected_files = st.multiselect(
                "Select documents to query:",
                options=[doc['filename'] for doc in stats["documents"]],
                default=[doc['filename'] for doc in stats["documents"]],
                help="Choose which documents to search in"
            )
        
        col3, col4, col5 = st.columns(3)
        with col3:
            content_type_filter = st.selectbox(
                "Content type:",
                options=["All", "text", "table", "image"],
                help="Filter by content type"
            )
        
        with col4:
            page_filter = st.number_input(
                "Specific page (optional):",
                min_value=0,
                value=0,
                help="Search in a specific page only"
            )
        
        if question and selected_files:
            with st.spinner("Generating answer..."):
                content_type = None if content_type_filter == "All" else content_type_filter
                page_num = None if page_filter == 0 else page_filter
                
                result = qa_system.ask_question(
                    question, 
                    selected_files, 
                    content_type=content_type,
                    page_num=page_num
                )
                
                st.subheader("Answer")
                confidence_color = "green" if result["confidence"] > 0.7 else "orange" if result["confidence"] > 0.4 else "red"
                st.markdown(f"**Confidence:** :{confidence_color}[{result['confidence']:.2f}]")
                
                st.write(result["answer"])
                
                if result["source_documents"]:
                    with st.expander("📚 Source References", expanded=True):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"📄 **File:** {doc.metadata.get('source', 'Unknown')}")
                            with col2:
                                st.write(f"📃 **Page:** {doc.metadata.get('page', 'Unknown')}")
                            with col3:
                                st.write(f"🏷️ **Type:** {doc.metadata.get('type', 'text').title()}")
                            
                            st.text_area(
                                f"Content {i}:",
                                value=doc.page_content,
                                height=150,
                                disabled=True,
                                key=f"source_{i}"
                            )
                            st.divider()
    
    with tab2:
        st.header("Browse Document Contents")
        
        stats = qa_system.get_document_stats()
        if stats["total_documents"] == 0:
            st.info("No documents to browse")
            return
        
        selected_doc = st.selectbox(
            "Select document:",
            options=[doc['filename'] for doc in stats["documents"]]
        )
        
        if selected_doc:
            doc_info = next(doc for doc in stats["documents"] if doc['filename'] == selected_doc)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Pages:** {doc_info['total_pages']}")
                st.write(f"**Total Chunks:** {doc_info['total_chunks']}")
            with col2:
                page_num = st.number_input(
                    "Select page:",
                    min_value=1,
                    max_value=doc_info['total_pages'],
                    value=1
                )
            
            if st.button("Get Document Summary"):
                with st.spinner("Generating summary..."):
                    summary = qa_system.get_document_summary(selected_doc)
                    st.subheader("Document Summary")
                    st.write(summary)
            
            st.subheader(f"Content from Page {page_num}")
            page_content = qa_system.get_document_content_by_page(selected_doc, page_num)
            
            if page_content:
                display_document_content(page_content)
            else:
                st.info(f"No content found for page {page_num}")
    
    with tab3:
        st.header("Search Documents")
        
        stats = qa_system.get_document_stats()
        if stats["total_documents"] == 0:
            st.info("No documents to search")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Search query:",
                placeholder="Enter keywords to search for...",
                help="Search across all document content"
            )
        
        with col2:
            search_files = st.multiselect(
                "Search in documents:",
                options=[doc['filename'] for doc in stats["documents"]],
                default=[doc['filename'] for doc in stats["documents"]]
            )
        
        if search_query and search_files:
            with st.spinner("Searching..."):
                search_results = qa_system.search_documents(search_query, search_files)
                
                if search_results:
                    st.subheader(f"Found {len(search_results)} results")
                    
                    for i, result in enumerate(search_results, 1):
                        with st.expander(f"Result {i} - {result['source']} (Page {result['page']}) - Score: {result['score']:.3f}"):
                            st.markdown(f"**Type:** {result['type'].title()}")
                            st.markdown(f"**Content:**")
                            st.text(result['content'])
                else:
                    st.info("No results found for your search query")
    
    with tab4:
        st.header("System Analytics")
        
        stats = qa_system.get_document_stats()
        
        if stats["total_documents"] > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Documents", stats["total_documents"])
            with col2:
                st.metric("Total Text Chunks", stats["total_chunks"])
            with col3:
                avg_chunks = stats["total_chunks"] / stats["total_documents"]
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
            
            st.subheader("Content Distribution")
            content_data = stats["content_types"]
            if content_data:
                import pandas as pd
                df = pd.DataFrame([
                    {"Content Type": k.title(), "Count": v} 
                    for k, v in content_data.items()
                ])
                st.bar_chart(df.set_index("Content Type"))
            
            st.subheader("Document Details")
            doc_data = []
            for doc in stats["documents"]:
                doc_data.append({
                    "Filename": doc["filename"],
                    "Pages": doc["total_pages"],
                    "Chunks": doc["total_chunks"],
                    "Processed": doc["processed_at"][:16]
                })
            
            if doc_data:
                import pandas as pd
                st.dataframe(pd.DataFrame(doc_data), use_container_width=True)
            
            if st.button("Export System Data"):
                export_data = qa_system.export_qa_history()
                st.download_button(
                    label="Download Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"qa_system_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No analytics data available. Upload documents to see system statistics.")
    
    st.divider()
    st.caption("Local PDF Q&A System - No internet required, all processing done locally")

if __name__ == "__main__":
    main()
