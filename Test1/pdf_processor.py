import pdfplumber
from typing import List, Dict, Any
from PIL import Image
import pytesseract
import cv2
import numpy as np

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        extracted_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    extracted_content.append({
                        'type': 'text',
                        'content': text,
                        'page': page_num,
                        'metadata': {'page': page_num, 'type': 'text'}
                    })
                
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table:
                        table_md = PDFProcessor._table_to_markdown(table)
                        extracted_content.append({
                            'type': 'table',
                            'content': table_md,
                            'page': page_num,
                            'metadata': {'page': page_num, 'type': 'table', 'table_index': table_idx}
                        })
        
        return extracted_content
    
    @staticmethod
    def _table_to_markdown(table: List[List[str]]) -> str:
        if not table or not table[0]:
            return ""
        
        header = "| " + " | ".join(str(cell) if cell else "" for cell in table[0]) + " |"
        separator = "| " + " | ".join("---" for _ in table[0]) + " |"
        
        rows = []
        for row in table[1:]:
            row_str = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
            rows.append(row_str)
        
        return "\n".join([header, separator] + rows)