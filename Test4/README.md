# Financial PDF Data Extractor with RAG

A comprehensive PDF processing system specialized for financial documents that combines multiple AI models and extraction techniques to enable intelligent querying of financial data through Retrieval-Augmented Generation (RAG).

## Overview
This system processes financial PDFs (annual reports, financial statements, sustainability reports) and enables natural language queries about the content. It handles both text-based and image-based tables, making it ideal for complex financial documents.

## Key Features
- **Multi-modal extraction**: Processes text, tables, and images from PDFs
- **Intelligent table extraction**: Uses both Tabula (text-based) and Tesseract OCR (image-based)
- **Advanced RAG implementation**: Semantic search with FAISS and transformer models
- **Financial-aware processing**: Optimized for financial terminology and numerical data
- **Confidence scoring**: Provides reliability metrics for extracted information

## Models Used
| Model | Purpose | Key Characteristics |
|-------|---------|---------------------|
| **SentenceTransformer (all-MiniLM-L6-v2)** | Generate semantic embeddings | 384-dimensional vectors, balanced performance/speed |
| **Google Flan-T5-base** | Text generation for complex queries | Text-to-text transformer, strong zero-shot capabilities |
| **RoBERTa-base-squad2** | Extractive question answering | Fine-tuned on SQuAD 2.0, high factual accuracy |

## Architecture
### PDF Processing Pipeline
```
PDF Input → Page Extraction → Multi-method Processing
                               ├── Text Extraction (PyMuPDF)
                               ├── Table Extraction (Tabula)
                               ├── Image Extraction (PyMuPDF)
                               └── OCR Processing (Tesseract)
```

### RAG Pipeline
```
User Query → Embedding Generation → FAISS Search → Context Retrieval
                                                       ↓
                                                 LLM Processing
                                                       ↓
                                                 Answer with Confidence
```

## Technical Components
### 1. Text Processing
- **RecursiveCharacterTextSplitter**: Chunks text (700 chars with 200 overlap)
- Financial context awareness: Identifies financial tables and metrics

### 2. Table Extraction
- **Tabula-py**: Extracts structured tables from text-based PDFs
- **OCR Pipeline**: OpenCV + Tesseract for image-based tables
- Preserves numerical formatting and alignment

### 3. Vector Database
- **FAISS** (Facebook AI Similarity Search)
- L2 distance metric for semantic similarity

### 4. Image Processing
- OpenCV for image preprocessing
- Base64 encoding for image storage

## Installation

### Prerequisites
- Python 3.8+
- Java JDK (for Tabula)
- Tesseract OCR

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/financial-pdf-extractor.git
cd financial-pdf-extractor

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
## Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
## Linux: sudo apt-get install tesseract-ocr
## Mac: brew install tesseract

# Set Java path (if needed)
export JAVA_HOME=/path/to/java
```

## Usage

### Basic Command
```bash
python financial_pdf_processor.py
```

### Menu Options
1. **Process new PDF**: 
   - Download from URL
   - Use local file
   - Select from common financial documents
   
2. **Query current PDF**: 
   - Natural language queries
   - Returns answers with confidence scores
   
3. **Exit**: Close application

### Example Queries
- "What was the revenue in 2022?"
- "Show me EBITDA for the last 3 years"
- "What are the total assets?"
- "Summarize the financial performance"
- "How many years of data does this report contain?"

### Output Structure
```
Answer: [Direct answer to your question]
Confidence: [High/Medium/Low] (X.X%)
Source pages: [Page numbers where information was found]
Additional context: [If available]
```

### Processed Data Structure
```
data_[filename]/
├── images/          # Extracted images
├── text/            # Text chunks
├── tables/          # Extracted tables (Tabula)
├── page_images/     # Full page renders
└── extracted_tables/ # OCR-extracted tables
```

## Performance Optimization
- **Embedding Generation**:
  - Batch processing
  - Progress tracking with tqdm
  - Selective processing (skip images if not needed)
  
- **Search Optimization**:
  - k=15 nearest neighbors
  - Relevance filtering based on query terms
  - Financial keyword prioritization

## Limitations
- ⚠️ OCR Accuracy: Complex table layouts may not extract perfectly
- ⚠️ Context Length: LLMs have token limits
- ⚠️ Image Embeddings: Currently skipped (text-only)
- ⚠️ Language: Optimized for English documents

## Future Enhancements
- 🌐 Multi-language support
- ☁️ Cloud storage integration
- 📦 Batch PDF processing
- 📊 Export to Excel/CSV
- 💻 Web interface
- 🚀 GPU acceleration for embeddings

## Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| Java not found | Set JAVA_HOME environment variable |
| Tesseract not found | Add Tesseract to system PATH |
| Memory errors | Reduce chunk_size or process smaller PDFs |
| Low confidence scores | Ensure PDF has clear text/tables |

### Enable Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing
Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Acknowledgments
- Hugging Face for transformer models
- Facebook Research for FAISS
- OpenAI for RAG concepts
- Tabula team for table extraction