import pymupdf
import tabula
import pandas as pd
from typing import List, Dict, Any

class PDFExtractor:
    def extract_content(self, filepath: str) -> List[Dict[str, Any]]:
        raw_content = []
        doc = pymupdf.open(filepath)

        print(f"Extracting content from {len(doc)} pages...")
        for page_num, page in enumerate(doc):
            page_content = {
                "page_number": page_num + 1,
                "text": "",
                "tables": [],
                "images": []
            }

            text = page.get_text()
            if text.strip():
                page_content["text"] = text

            try:
                tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True, lattice=True)
                if tables:
                    for table_df in tables:
                        if not table_df.empty:
                            page_content["tables"].append(table_df)
            except Exception as e:
                print(f"Warning: Tabula failed on page {page_num + 1}. Error: {e}")

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = pymupdf.Pixmap(doc, xref)
                img_bytes = pix.tobytes("png")
                
                bbox = None
                if len(img) > 1:
                    bbox = page.get_image_bbox(img)
                
                if img_bytes:
                    page_content["images"].append({
                        "data": img_bytes,
                        "index": img_index,
                        "width": pix.width,
                        "height": pix.height,
                        "bbox": bbox
                    })
                pix = None

            raw_content.append(page_content)
        
        doc.close()
        print(f"✅ Extracted content from {len(raw_content)} pages.")
        return raw_content