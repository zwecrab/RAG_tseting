import pandas as pd
import json
import re
from typing import List, Dict, Any
from services import ImageToTextService

class ContentProcessor:
    def __init__(self, image_service: ImageToTextService):
        self.image_service = image_service

    def process_content(self, raw_content: List[Dict], source_pdf_name: str):
        page_data_list = []
        all_images = []
        
        print(f"Processing {len(raw_content)} pages...")

        for page_content in raw_content:
            page_number = page_content["page_number"]
            
            html_content, json_metadata, page_summary, images_to_store = self._process_page(
                page_content, source_pdf_name
            )
            
            page_data = {
                "source_pdf": source_pdf_name,
                "page_number": page_number,
                "html_content": html_content,
                "json_metadata": json_metadata,
                "page_summary": page_summary
            }
            
            page_data_list.append(page_data)
            all_images.extend(images_to_store)

        print(f"✅ Processed {len(page_data_list)} pages and {len(all_images)} images.")
        return page_data_list, all_images

    def _process_page(self, page_content: Dict, source_pdf_name: str):
        page_number = page_content["page_number"]
        text = page_content.get("text", "")
        tables = page_content.get("tables", [])
        images = page_content.get("images", [])
        
        html_parts = []
        json_elements = []
        summary_parts = []
        images_to_store = []
        
        table_names = self._extract_table_names_from_text(text)
        
        if text.strip():
            cleaned_text = self._clean_text_for_html(text)
            html_parts.append(f"<div class='text-content'>{self._escape_html(cleaned_text)}</div>")
            json_elements.append({
                "type": "text",
                "content": cleaned_text,
                "length": len(cleaned_text),
                "table_references": table_names
            })
            summary_parts.append(cleaned_text[:500])

        for table_idx, table_df in enumerate(tables):
            table_name = table_names[table_idx] if table_idx < len(table_names) else f"Table {table_idx + 1}"
            
            table_html = self._create_structured_table_html(table_df, table_name)
            table_json = self._create_structured_table_json(table_df, table_name)
            table_text = self._create_searchable_table_text(table_df, table_name)
            
            html_parts.append(f"<div class='table-content' id='table-{table_idx}' data-table-name='{table_name}'>{table_html}</div>")
            
            json_elements.append({
                "type": "table",
                "index": table_idx,
                "table_name": table_name,
                "rows": len(table_df),
                "columns": len(table_df.columns),
                "structure": table_json,
                "searchable_text": table_text
            })
            summary_parts.append(f"{table_name}: {table_text[:300]}")

        for img_idx, img_data in enumerate(images):
            image_type = self._classify_image(img_data["data"])
            
            if image_type == "text_table":
                print(f"  > Processing text/table image from page {page_number}...")
                image_text = self.image_service.get_text_from_image(img_data["data"])
                if image_text:
                    html_parts.append(f"<div class='image-text-content' id='image-{img_idx}'>{self._escape_html(image_text)}</div>")
                    json_elements.append({
                        "type": "image_text",
                        "index": img_idx,
                        "content": image_text,
                        "width": img_data.get("width"),
                        "height": img_data.get("height")
                    })
                    summary_parts.append(image_text[:300])
            else:
                bbox = img_data.get("bbox")
                position_x, position_y = (bbox[0], bbox[1]) if bbox else (None, None)
                
                images_to_store.append({
                    "source_pdf": source_pdf_name,
                    "page_number": page_number,
                    "image_index": img_idx,
                    "image_data": img_data["data"],
                    "image_format": "png",
                    "image_type": image_type,
                    "width": img_data.get("width"),
                    "height": img_data.get("height"),
                    "position_x": position_x,
                    "position_y": position_y
                })
                
                html_parts.append(f"<div class='image-placeholder' id='image-{img_idx}' data-type='{image_type}'>Image: {image_type}</div>")
                json_elements.append({
                    "type": "image_placeholder",
                    "index": img_idx,
                    "image_type": image_type,
                    "width": img_data.get("width"),
                    "height": img_data.get("height"),
                    "position_x": position_x,
                    "position_y": position_y
                })

        html_content = f"<div class='page' data-page='{page_number}'>{''.join(html_parts)}</div>"
        
        json_metadata = {
            "page_number": page_number,
            "content_types": list(set([elem["type"] for elem in json_elements])),
            "elements": json_elements,
            "word_count": len(text.split()) if text else 0,
            "has_images": len([e for e in json_elements if "image" in e["type"]]) > 0,
            "has_tables": len([e for e in json_elements if e["type"] == "table"]) > 0,
            "table_names": table_names
        }
        
        page_summary = " ".join(summary_parts)[:2000]
        
        return html_content, json_metadata, page_summary, images_to_store

    def _extract_table_names_from_text(self, text: str) -> List[str]:
        table_pattern = r'Table\s+(\d+(?:\.\d+)?(?:[a-zA-Z])?)\s*:?\s*([^\n]*)'
        matches = re.findall(table_pattern, text, re.IGNORECASE)
        table_names = []
        for match in matches:
            table_num, table_desc = match
            table_name = f"Table {table_num}"
            if table_desc.strip():
                table_name += f": {table_desc.strip()}"
            table_names.append(table_name)
        return table_names

    def _clean_text_for_html(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not self._is_likely_table_data(line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _is_likely_table_data(self, line: str) -> bool:
        words = line.split()
        if len(words) < 2:
            return False
        
        numeric_count = sum(1 for word in words if re.match(r'^[\d,.\(\)]+$', word))
        return numeric_count / len(words) > 0.4

    def _create_structured_table_html(self, df: pd.DataFrame, table_name: str) -> str:
        df_clean = df.copy()
        df_clean = df_clean.fillna('')
        
        for col in df_clean.columns:
            if 'Unnamed' in str(col):
                if col == df_clean.columns[0]:
                    df_clean = df_clean.rename(columns={col: 'Item'})
                else:
                    df_clean = df_clean.rename(columns={col: f'Value_{col.split(":")[-1] if ":" in str(col) else "1"}'})
        
        html = f"<div class='table-header'><h4>{table_name}</h4></div>"
        html += df_clean.to_html(classes="table table-striped table-structured", escape=False, index=False)
        return html

    def _create_structured_table_json(self, df: pd.DataFrame, table_name: str) -> Dict:
        df_clean = df.fillna('')
        
        headers = []
        for col in df.columns:
            if 'Unnamed' in str(col):
                if col == df.columns[0]:
                    headers.append('Item')
                else:
                    headers.append(f'Value_{col.split(":")[-1] if ":" in str(col) else "1"}')
            else:
                headers.append(str(col))
        
        rows = []
        for _, row in df_clean.iterrows():
            row_dict = {}
            for i, (original_col, clean_header) in enumerate(zip(df.columns, headers)):
                row_dict[clean_header] = str(row[original_col]) if pd.notna(row[original_col]) else ""
            rows.append(row_dict)
        
        return {
            "table_name": table_name,
            "headers": headers,
            "rows": rows,
            "dimensions": {
                "rows": len(df),
                "columns": len(df.columns)
            }
        }

    def _create_searchable_table_text(self, df: pd.DataFrame, table_name: str) -> str:
        df_clean = df.fillna('')
        
        text_parts = [table_name]
        
        headers = [str(col) for col in df.columns]
        if headers:
            text_parts.append("Headers: " + ", ".join(headers))
        
        for _, row in df_clean.iterrows():
            row_text = []
            for col, value in zip(df.columns, row):
                if pd.notna(value) and str(value).strip():
                    clean_col = str(col).replace('Unnamed: ', '').replace(':', '')
                    row_text.append(f"{clean_col}: {value}")
            if row_text:
                text_parts.append(" | ".join(row_text))
        
        return ". ".join(text_parts)

    def _classify_image(self, image_bytes: bytes) -> str:
        try:
            ocr_result = self.image_service.get_text_from_image(image_bytes)
            if self._has_substantial_text_or_table(ocr_result):
                return "text_table"
            else:
                return self._classify_non_text_image(image_bytes)
        except:
            return "other"

    def _has_substantial_text_or_table(self, ocr_text: str) -> bool:
        if not ocr_text or len(ocr_text.strip()) < 10:
            return False
        
        words = ocr_text.split()
        if len(words) < 5:
            return False
            
        table_indicators = ["|", "─", "│", "┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼"]
        has_table_chars = any(char in ocr_text for char in table_indicators)
        
        lines = ocr_text.split('\n')
        has_multiple_lines = len([line for line in lines if line.strip()]) > 2
        
        return has_table_chars or has_multiple_lines

    def _classify_non_text_image(self, image_bytes: bytes) -> str:
        return "chart"

    def _escape_html(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")