# Local PDF Q&A System

A comprehensive PDF Question & Answer system that runs entirely on your local machine without requiring any API keys. The system supports text extraction, table processing, image OCR, and intelligent question answering using local AI models.

## Features

### 📚 Complete PDF Processing
- **Text Extraction**: Extract text from PDFs using multiple methods (pdfplumber + PyMuPDF)
- **Table Processing**: Detect and extract tables with intelligent formatting
- **Image OCR**: Extract text from images within PDFs using Tesseract
- **Multi-format Support**: Handles complex PDFs with mixed content types

### 🤖 Local AI Processing
- **No API Keys Required**: All processing done locally
- **Hybrid Search**: Combines dense vector search with BM25 sparse retrieval
- **Local Language Models**: Uses Hugging Face transformers (FLAN-T5)
- **Smart Chunking**: Intelligent text segmentation preserving context

### 🔍 Advanced Query Features
- **Multi-document Search**: Query across multiple PDFs simultaneously
- **Content Type Filtering**: Search specifically in text, tables, or images
- **Page-specific Queries**: Target specific pages for focused answers
- **Confidence Scoring**: Get reliability scores for generated answers

### 📊 Rich Interface
- **Document Browser**: Navigate through document contents page by page
- **Search Interface**: Semantic search across all loaded documents
- **Analytics Dashboard**: View document statistics and content distribution
- **Export Capabilities**: Export system data and search histories

## Installation

### Prerequisites
1. **Python 3.8+**
2. **Tesseract OCR** (for image text extraction)
   - Windows: Download from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Linux: `sudo apt-get install tesseract-ocr`

### Setup
1. **Clone or download all files to a directory**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser to:** `http://localhost:8501`

## File Structure

```
pdf-qa-system/
├── main.py                 # Application launcher
├── streamlit_app.py        # Streamlit web interface
├── requirements.txt        # Python dependencies
├── config.py              # Configuration settings
├── pdf_processor.py       # PDF content extraction
├── text_processing.py     # Text processing and chunking
├── vector_store.py        # Vector storage and retrieval
├── local_llm.py          # Local language model interface
├── qa_system.py          # Main Q&A system orchestrator
├── uploaded_pdfs/        # Uploaded PDF storage
├── vector_stores/        # Vector database storage
├── temp/                 # Temporary processing files
└── extracted_images/     # Extracted images from PDFs
```

## Usage Guide

### 1. Upload Documents
- Use the sidebar to upload PDF files
- System automatically processes text, tables, and images
- View processing progress and statistics

### 2. Ask Questions
- Enter questions in natural language
- Select specific documents to query
- Filter by content type (text/table/image)
- Get answers with confidence scores and source references

### 3. Browse Documents
- Navigate through documents page by page
- View extracted content including tables and images
- Generate document summaries

### 4. Search