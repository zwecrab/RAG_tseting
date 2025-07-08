import pymupdf
import tabula
import pandas as pd
from typing import List, Dict, Any

class PDFExtractor:
    """
    Extracts raw content (text, tables as DataFrames, and image bytes) from a PDF file.
    """
    def extract_content(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Extracts all content from a PDF, page by page.
        """
        raw_content = []
        doc = pymupdf.open(filepath)

        print(f"Extracting content from {len(doc)} pages...")
        for page_num, page in enumerate(doc):
            # 1. Extract plain text
            text = page.get_text()
            if text.strip():
                raw_content.append({
                    "type": "text",
                    "data": text,
                    "page_number": page_num + 1
                })

            # 2. Extract tables using Tabula
            try:
                tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True, lattice=True)
                if tables:
                    for table_df in tables:
                        if not table_df.empty:
                            raw_content.append({
                                "type": "table",
                                "data": table_df,
                                "page_number": page_num + 1
                            })
            except Exception as e:
                print(f"Warning: Tabula failed on page {page_num + 1}. It might not contain text-based tables. Error: {e}")

            # 3. Extract images as bytes for vision model processing
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                img_bytes = pix.tobytes("png")  # Extract as bytes
                if img_bytes:
                    raw_content.append({
                        "type": "image",
                        "data": img_bytes,
                        "page_number": page_num + 1,
                        "img_index": img_index
                    })
        
        doc.close()
        print(f"✅ Extracted {len(raw_content)} raw content items.")
        return raw_content
