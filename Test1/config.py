import os
from dataclasses import dataclass

@dataclass
class Config:
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "microsoft/DialoGPT-medium"
    UPLOAD_DIR: str = "uploaded_pdfs"
    VECTOR_STORE_DIR: str = "vector_stores"
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.1
    TOP_K: int = 3
    
    def __post_init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)