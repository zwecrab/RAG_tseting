import fitz
import pdfplumber
import pytesseract
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile
import io
from config import Config

class PDFProcessor:
    def __init__(self):
        self.config = Config()
        
    def extract_content_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        extracted_content = []
        
        try:
            extracted_content.extend(self._extract_with_pdfplumber(pdf_path))
            extracted_content.extend(self._extract_with_pymupdf(pdf_path))
            extracted_content = self._deduplicate_content(extracted_content)
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {str(e)}")
            
        return extracted_content
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text and text.strip():
                    content.append({
                        'type': 'text',
                        'content': text.strip(),
                        'page': page_num,
                        'source': 'pdfplumber',
                        'metadata': {
                            'page': page_num,
                            'type': 'text',
                            'extractor': 'pdfplumber'
                        }
                    })
                
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:
                        table_content = self._process_table(table, page_num, table_idx)
                        if table_content:
                            content.append(table_content)
        
        return content
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        content = []
        
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            text = page.get_text()
            if text and text.strip():
                content.append({
                    'type': 'text',
                    'content': text.strip(),
                    'page': page_num + 1,
                    'source': 'pymupdf',
                    'metadata': {
                        'page': page_num + 1,
                        'type': 'text',
                        'extractor': 'pymupdf'
                    }
                })
            
            images = self._extract_images_from_page(page, page_num + 1)
            content.extend(images)
            
        doc.close()
        return content
    
    def _extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        images_content = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)
                
                if pix.n - pix.alpha < 4:
                    img_data = pix.tobytes("png")
                    img_path = self.config.EXTRACTED_IMAGES_DIR / f"page_{page_num}_img_{img_index}.png"
                    
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    ocr_text = self._extract_text_from_image(img_path)
                    
                    images_content.append({
                        'type': 'image',
                        'content': ocr_text if ocr_text else f"[Image {img_index + 1} on page {page_num}]",
                        'page': page_num,
                        'image_path': str(img_path),
                        'metadata': {
                            'page': page_num,
                            'type': 'image',
                            'image_index': img_index,
                            'has_text': bool(ocr_text)
                        }
                    })
                
                pix = None
                
            except Exception as e:
                print(f"Error extracting image {img_index} from page {page_num}: {str(e)}")
                continue
        
        return images_content
    
    def _extract_text_from_image(self, image_path: Path) -> str:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return ""
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            kernel = np.ones((1, 1), np.uint8)
            img_processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            img_processed = cv2.medianBlur(img_processed, 3)
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?-'
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            text = pytesseract.image_to_string(img_processed, config=custom_config)
            
            return text.strip()
            
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return ""
    
    def _process_table(self, table: List[List[str]], page_num: int, table_idx: int) -> Dict[str, Any]:
        try:
            df = pd.DataFrame(table[1:], columns=table[0])
            df = df.dropna(how='all').fillna('')
            
            if df.empty:
                return None
            
            table_markdown = self._dataframe_to_markdown(df)
            table_text = self._dataframe_to_text(df)
            
            return {
                'type': 'table',
                'content': f"Table {table_idx + 1} from page {page_num}:\n\n{table_markdown}\n\nText format:\n{table_text}",
                'page': page_num,
                'table_data': df.to_dict('records'),
                'metadata': {
                    'page': page_num,
                    'type': 'table',
                    'table_index': table_idx,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
        except Exception as e:
            print(f"Error processing table: {str(e)}")
            return None
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        return df.to_markdown(index=False, tablefmt='pipe')
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        text_parts = []
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
            text_parts.append(row_text)
        return "\n".join(text_parts)
    
    def _deduplicate_content(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_content = set()
        deduplicated = []
        
        for item in content:
            content_hash = hash(item['content'][:200])
            key = (item['page'], item['type'], content_hash)
            
            if key not in seen_content:
                seen_content.add(key)
                deduplicated.append(item)
        
        return deduplicated
