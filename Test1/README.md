# Local PDF Q&A System

A lightweight PDF question-answering system that runs entirely on your laptop without requiring external API keys. Uses local AI models for document processing and question answering.

## Features

- **No API Keys Required**: Uses local transformer models
- **Laptop Optimized**: Lightweight models that run on CPU
- **Hybrid Search**: Combines dense embeddings with BM25 for better retrieval
- **Table Support**: Extracts and processes tables from PDFs
- **Multi-PDF Queries**: Query across multiple documents simultaneously
- **Real-time Processing**: Instant document upload and processing

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract (optional, for OCR):**
   - **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt install tesseract-ocr`

## Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Upload PDFs:**
   - Use the sidebar to upload one or more PDF files
   - Wait for processing (first time may take longer as models download)

3. **Ask questions:**
   - Select which PDFs to query
   - Type your question in the text input
   - Get answers with source references

## File Structure

```
pdf-qa-system/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── pdf_processor.py      # PDF text and table extraction
├── chunker.py           # Smart text chunking
├── retriever.py         # Hybrid retrieval system
├── llm_handler.py       # Local language model handler
├── document_processor.py # Document processing utilities
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── uploaded_pdfs/       # Created automatically for uploads
└── vector_stores/       # Created automatically for embeddings
```

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models (downloaded automatically)
- **CPU**: Any modern processor (GPU optional but not required)

## Models Used

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (22MB)
- **Language Model**: distilbert-base-cased-distilled-squad (250MB)
- **Fallback**: Simple keyword-based extraction for maximum compatibility

## Performance Tips

1. **First Run**: Initial model download may take a few minutes
2. **Memory**: Close other applications for better performance
3. **File Size**: Larger PDFs take longer to process
4. **Queries**: Specific questions work better than general ones

## Troubleshooting

**Models not loading:**
- Check internet connection for initial download
- Ensure sufficient disk space (2GB)

**PDF processing errors:**
- Ensure PDFs are not password-protected
- Some scanned PDFs may not extract text properly

**Slow performance:**
- Reduce number of uploaded PDFs
- Use smaller chunk sizes in config.py
- Close other resource-intensive applications

## Customization

Edit `config.py` to adjust:
- Chunk size and overlap
- Model selection
- Directory paths
- Generation parameters

## License

Open source - feel free to modify and distribute.

## Contributing

Submit issues and pull requests on the project repository.