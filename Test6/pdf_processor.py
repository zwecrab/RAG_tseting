# pdf_processor.py

import pdfplumber
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'<[^>]+>', '', text).strip()

def extract_text_from_pdf(pdf_path):
    print(f"Processing PDF with pdfplumber: {pdf_path}...")
    all_text_chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"  - Reading page {page_num + 1}...")

            # 1. Extract plain text and split it into smaller lines
            plain_text = page.extract_text()
            if plain_text:
                # Split by newline and add each non-empty line as a separate chunk
                lines = [line.strip() for line in plain_text.split('\n') if len(line.strip()) > 10]
                all_text_chunks.extend(lines)

            # 2. Extract and CLEAN tables
            tables = page.extract_tables()
            if tables:
                print(f"  - Found {len(tables)} table(s) on page {page_num + 1}.")
                for table in tables:
                    if not table or not table[0]: continue
                    header = [clean_text(cell) for cell in table[0]]
                    for row in table[1:]:
                        row_data = [clean_text(cell) for cell in row if cell is not None]
                        row_text = ", ".join(f"{h}: {c}" for h, c in zip(header, row_data) if c and h)
                        if row_text:
                            all_text_chunks.append(f"Table data: {row_text}")

            # 3. OCR for images
            for img_obj in page.images:
                try:
                    img = img_obj.to_image(resolution=200)
                    print(f"  - OCR on image on page {page_num + 1}...")
                    ocr_text = pytesseract.image_to_string(img.original)
                    if ocr_text.strip():
                        # Split OCR results into lines as well
                        ocr_lines = [line.strip() for line in ocr_text.split('\n') if len(line.strip()) > 10]
                        all_text_chunks.extend(ocr_lines)
                except Exception as e:
                    print(f"    Could not process image on page {page_num + 1}: {e}")
    
    # The function now returns a list of focused chunks
    return all_text_chunks