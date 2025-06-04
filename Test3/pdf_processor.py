import fitz
import tabula
import subprocess

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF as a list of paragraphs."""
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                if text:
                    paragraphs.append(text)
    doc.close()
    return paragraphs

def extract_images_from_pdf(pdf_path):
    """Extract images from a PDF as a list of byte objects."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(image_bytes)
    doc.close()
    return images

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF using tabula-py with custom encoding."""
    try:
        # Try default UTF-8 encoding
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        return tables
    except UnicodeDecodeError:
        # Fallback to reading with a different encoding (e.g., Windows-1252)
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, encoding='latin1')
        return tables