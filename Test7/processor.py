import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from services import ImageToTextService

class ContentProcessor:
    """
    Processes raw extracted content into clean, embeddable text chunks.
    """
    def __init__(self, image_service: ImageToTextService, chunk_size: int, chunk_overlap: int):
        self.image_service = image_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def process_content(self, raw_content: List[Dict], source_pdf_name: str) -> List[Dict]:
        """
        Iterates through raw content and processes it based on its type.
        """
        processed_chunks = []
        print(f"Processing {len(raw_content)} raw items...")

        for item in raw_content:
            item_type = item.get("type")
            page_num = item.get("page_number")
            data = item.get("data")

            if item_type == "text":
                chunks = self.text_splitter.split_text(data)
                for chunk in chunks:
                    processed_chunks.append({
                        "source_pdf": source_pdf_name,
                        "page_number": page_num,
                        "chunk_type": "text",
                        "content_text": chunk
                    })

            elif item_type == "table":
                table_text = self._format_dataframe_to_text(data)
                if table_text:
                    processed_chunks.append({
                        "source_pdf": source_pdf_name,
                        "page_number": page_num,
                        "chunk_type": "table",
                        "content_text": f"Table Data: {table_text}"
                    })

            elif item_type == "image":
                print(f"  > Processing image from page {page_num} with vision model...")
                image_text = self.image_service.get_text_from_image(data)
                if image_text:
                    # Further chunk the text from the image
                    img_text_chunks = self.text_splitter.split_text(image_text)
                    for chunk in img_text_chunks:
                        processed_chunks.append({
                            "source_pdf": source_pdf_name,
                            "page_number": page_num,
                            "chunk_type": "image_text",
                            "content_text": f"Text from image: {chunk}"
                        })

        print(f"✅ Processed content into {len(processed_chunks)} final chunks.")
        return processed_chunks

    def _format_dataframe_to_text(self, df: pd.DataFrame) -> str:
        """
        Converts a pandas DataFrame into a clean, readable string.
        """
        if not isinstance(df, pd.DataFrame):
            return ""
        
        # Replace NaN with empty strings
        df = df.fillna('')
        
        # Convert all data to string type
        df = df.astype(str)

        # Create a descriptive string for each row
        row_strings = []
        header = df.columns.tolist()
        for index, row in df.iterrows():
            row_desc = ", ".join([f"{header[i]}: {val}" for i, val in enumerate(row) if val])
            if row_desc:
                row_strings.append(row_desc)
        
        return "; ".join(row_strings)
