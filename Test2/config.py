import os
from pathlib import Path

class Config:
    BASE_DIR = Path(__file__).parent
    UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
    VECTOR_STORE_DIR = BASE_DIR / "vector_stores"
    TEMP_DIR = BASE_DIR / "temp"
    EXTRACTED_IMAGES_DIR = BASE_DIR / "extracted_images"
    
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 64
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "microsoft/DialoGPT-medium"
    
    IMAGE_DPI = 300
    IMAGE_MAX_SIZE = (1024, 1024)
    
    SUPPORTED_FORMATS = ['.pdf']
    
    MAX_FILE_SIZE_MB = 100
    MAX_TOTAL_FILES = 20
    
    RETRIEVER_TOP_K = 5
    BM25_WEIGHT = 0.3
    VECTOR_WEIGHT = 0.7
    
    @classmethod
    def create_directories(cls):
        for dir_path in [cls.UPLOAD_DIR, cls.VECTOR_STORE_DIR, cls.TEMP_DIR, cls.EXTRACTED_IMAGES_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
