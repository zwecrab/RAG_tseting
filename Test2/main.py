#!/usr/bin/env python3

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    try:
        import streamlit
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import pdfplumber
        import fitz
        import pytesseract
        import cv2
        import numpy
        import pandas
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False
    return True

def check_tesseract():
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        print("Tesseract OCR not found. Please install Tesseract:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- macOS: brew install tesseract")
        print("- Linux: sudo apt-get install tesseract-ocr")
        return False

def setup_directories():
    from config import Config
    Config.create_directories()
    print("Created necessary directories")

def main():
    import pytesseract
    
    print("🚀 Starting Local PDF Q&A System...")
    
    if not check_dependencies():
        sys.exit(1)
    
    if not check_tesseract():
        print("⚠️  Warning: Tesseract not found. OCR functionality will be limited.")
    
    setup_directories()
    
    print("✅ All checks passed!")
    print("🌐 Starting Streamlit server...")
    
    streamlit_path = Path(__file__).parent / "streamlit_app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down PDF Q&A System...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
