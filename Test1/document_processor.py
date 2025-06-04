import os
from typing import Tuple, List
from langchain.schema import Document
from pdf_processor import PDFProcessor
from chunker import SmartChunker
from config import Config

def process_pdf(file_path: str, file_name: str) -> Tuple[str, List[Document]]:
    try:
        pdf_processor = PDFProcessor()
        extracted_content = pdf_processor.extract_text_from_pdf(file_path)
        
        chunker = SmartChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        documents = chunker.chunk_content(extracted_content)
        
        for doc in documents:
            doc.metadata['source'] = file_name
        
        return "success", documents
    except Exception as e:
        return f"error: {str(e)}", []

def save_uploaded_file(uploaded_file) -> str:
    config = Config()
    file_path = os.path.join(config.UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path