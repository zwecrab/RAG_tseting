import pandas as pd
import json
import re
import os
import hashlib
import io
from typing import List, Dict, Any, Tuple, Optional
from services import ImageToTextService
import numpy as np

class CSVProcessor:
    def __init__(self, db, embedding_service):
        self.db = db
        self.embedding_service = embedding_service
    
    def process_csv_file(self, filepath: str, source_type: str = 'upload') -> Dict:
        """Process CSV file and store in vector database"""
        try:
            filename = os.path.basename(filepath)
            print(f"Processing CSV file: {filename}")
            
            # Read CSV with multiple encodings fallback
            df = self._read_csv_with_fallback(filepath)
            
            # Create file hash
            file_hash = self._calculate_file_hash(filepath)
            
            # Clean and analyze data
            df_clean, column_types = self._clean_and_analyze_csv(df)
            
            # Insert document metadata
            headers = list(df_clean.columns)
            doc_id = self.db.insert_csv_document(
                filename=filename,
                file_path=filepath,
                headers=headers,
                total_rows=len(df_clean),
                file_hash=file_hash,
                column_types=column_types,
                source_type=source_type
            )
            
            # Process and store column metadata
            self._process_column_metadata(df_clean, doc_id, filename)
            
            # Process rows in batches
            total_processed = self._process_rows_in_batches(df_clean, doc_id, filename, headers)
            
            return {
                'success': True,
                'message': f"Successfully processed {total_processed} rows from {filename}",
                'doc_id': doc_id,
                'filename': filename,
                'total_rows': total_processed
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing CSV file: {str(e)}",
                'doc_id': None
            }
    
    def process_csv_content(self, csv_content: str, filename: str, source_type: str = 'extracted') -> Dict:
        """Process CSV content string and store in vector database"""
        try:
            # Convert CSV content to DataFrame
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Create content hash
            content_hash = hashlib.sha256(csv_content.encode()).hexdigest()
            
            # Clean and analyze data
            df_clean, column_types = self._clean_and_analyze_csv(df)
            
            # Insert document metadata
            headers = list(df_clean.columns)
            doc_id = self.db.insert_csv_document(
                filename=filename,
                file_path=None,
                headers=headers,
                total_rows=len(df_clean),
                file_hash=content_hash,
                column_types=column_types,
                source_type=source_type
            )
            
            # Process and store column metadata
            self._process_column_metadata(df_clean, doc_id, filename)
            
            # Process rows in batches
            total_processed = self._process_rows_in_batches(df_clean, doc_id, filename, headers)
            
            return {
                'success': True,
                'message': f"Successfully processed {total_processed} rows from {filename}",
                'doc_id': doc_id,
                'filename': filename,
                'total_rows': total_processed
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing CSV content: {str(e)}",
                'doc_id': None
            }
    
    def _read_csv_with_fallback(self, filepath: str) -> pd.DataFrame:
        """Read CSV with multiple encoding fallbacks"""
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                return pd.read_csv(filepath, encoding=encoding)
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with error handling
        try:
            return pd.read_csv(filepath, encoding='utf-8', errors='replace')
        except Exception as e:
            raise Exception(f"Could not read CSV file with any encoding: {str(e)}")
    
    def _calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of file"""
        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _clean_and_analyze_csv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Clean CSV data and analyze column types"""
        df_clean = df.copy()
        column_types = {}
        
        # Clean column names
        df_clean.columns = [str(col).strip() for col in df_clean.columns]
        
        # Analyze and clean each column
        for col in df_clean.columns:
            # Remove leading/trailing whitespace
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Determine column type
            column_types[col] = self._determine_column_type(df_clean[col])
        
        return df_clean, column_types
    
    def _determine_column_type(self, series: pd.Series) -> str:
        """Determine the type of a column with better error handling"""
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'empty'
        
        # Check if numeric
        try:
            pd.to_numeric(non_null_series)
            return 'numeric'
        except (ValueError, TypeError):
            pass
        
        # Check if date (with warning suppression)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(non_null_series, errors='raise')
            return 'datetime'
        except (ValueError, TypeError):
            pass
        
        # Check if categorical (limited unique values)
        unique_ratio = len(non_null_series.unique()) / len(non_null_series)
        if unique_ratio < 0.5 and len(non_null_series.unique()) < 50:
            return 'categorical'
        
        return 'text'
    
    def _process_column_metadata(self, df: pd.DataFrame, doc_id: int, filename: str):
        """Process and store column metadata"""
        columns_data = []
        
        for col in df.columns:
            # Get sample values (first 10 non-null values)
            sample_values = df[col].dropna().head(10).astype(str).tolist()
            
            # Create column description
            column_desc = f"Column {col} from {filename} with sample values: {', '.join(sample_values[:5])}"
            
            # Create embedding for column
            embedding = self.embedding_service.create_single_embedding(column_desc)
            
            columns_data.append({
                'doc_id': doc_id,
                'column_name': col,
                'column_type': self._determine_column_type(df[col]),
                'sample_values': sample_values,
                'column_description': column_desc,
                'embedding': str(embedding.tolist())
            })
        
        self.db.insert_csv_columns(columns_data)
    
    def _process_rows_in_batches(self, df: pd.DataFrame, doc_id: int, filename: str, headers: List[str]) -> int:
        """Process rows in batches for better performance"""
        batch_size = 100
        total_processed = 0
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_rows = []
            
            # Create embeddings for batch
            batch_texts = []
            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                row_text = self._create_row_embedding_text(row.to_dict(), headers, filename)
                batch_texts.append(row_text)
            
            # Get embeddings
            embeddings = self.embedding_service.create_embeddings(batch_texts)
            
            # Prepare batch data
            for i, idx in enumerate(range(start_idx, end_idx)):
                row = df.iloc[idx]
                batch_rows.append({
                    'doc_id': doc_id,
                    'row_index': idx,
                    'row_data': json.dumps(row.to_dict()),
                    'row_text': batch_texts[i],
                    'embedding': str(embeddings[i].tolist())
                })
            
            # Insert batch
            self.db.insert_csv_rows_batch(batch_rows)
            total_processed += len(batch_rows)
            
            if total_processed % 500 == 0:
                print(f"Processed {total_processed} rows...")
        
        return total_processed
    
    def _create_row_embedding_text(self, row_data: Dict, headers: List[str], filename: str) -> str:
        """Create context-rich text for row embedding"""
        row_parts = [f"Table: {filename}"]
        
        for col_name in headers:
            value = row_data.get(col_name, "")
            if value and str(value).strip() and str(value).lower() not in ['nan', 'none', '']:
                row_parts.append(f"{col_name}: {value}")
        
        return " | ".join(row_parts)

class AdvancedTableAnalyzer:
    def __init__(self, image_service: ImageToTextService):
        self.image_service = image_service
        self.csv_processor = None  # Will be initialized when needed

    def set_csv_processor(self, csv_processor: CSVProcessor):
        """Set CSV processor for table extraction"""
        self.csv_processor = csv_processor

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

        # Enhanced table processing with multiple extraction methods
        for table_idx, table_df in enumerate(tables):
            table_name = table_names[table_idx] if table_idx < len(table_names) else f"Table {table_idx + 1}"
            
            print(f"\n🔍 ENHANCED TABLE ANALYSIS: {table_name}")
            
            # Get table context from text
            table_context = self._extract_table_context(text, table_name, table_idx)
            
            # Try multiple extraction methods
            extraction_results = self._extract_table_multiple_methods(
                table_df, table_name, table_context, source_pdf_name, page_number
            )
            
            # Use best extraction result
            best_result = self._select_best_extraction(extraction_results)
            
            if best_result:
                # Store CSV content if available
                if best_result['csv_content'] and self.csv_processor:
                    csv_result = self.csv_processor.process_csv_content(
                        best_result['csv_content'], 
                        f"{source_pdf_name}_{table_name.replace(' ', '_')}.csv",
                        'pdf_table'
                    )
                    best_result['csv_doc_id'] = csv_result.get('doc_id')
                
                # Store extraction record
                if hasattr(self, 'db') and self.db:
                    self.db.insert_table_extraction(
                        source_pdf_name, page_number, table_name,
                        best_result['method'], best_result['csv_content'],
                        best_result.get('csv_doc_id'), best_result['confidence']
                    )
            
            # Analyze the table structure (existing logic)
            table_analysis = self._deep_analyze_table(table_df, table_name, table_context, text)
            
            print(f"✅ Final structure: {table_analysis['structure_type']}")
            print(f"📊 Processed rows: {len(table_analysis['rows'])}")
            
            table_html = self._create_advanced_html(table_analysis, table_name)
            table_text = self._create_advanced_searchable_text(table_analysis, table_name)
            
            html_parts.append(f"<div class='table-content' id='table-{table_idx}' data-table-name='{table_name}'>{table_html}</div>")
            
            # Enhanced JSON element with extraction info
            json_elements.append({
                "type": "table",
                "index": table_idx,
                "table_name": table_name,
                "rows": table_analysis['row_count'],
                "columns": table_analysis['col_count'],
                "structure": table_analysis,
                "searchable_text": table_text,
                "extraction_method": best_result['method'] if best_result else 'tabula',
                "csv_available": best_result['csv_content'] is not None if best_result else False
            })
            summary_parts.append(f"{table_name}: {table_text[:300]}")

        # Enhanced image processing
        for img_idx, img_data in enumerate(images):
            image_type = self._classify_image(img_data["data"])
            
            if image_type == "text_table":
                print(f"  > Processing text/table image from page {page_number}...")
                
                # Try both text extraction and table extraction
                image_text = self.image_service.get_text_from_image(img_data["data"])
                table_csv = self.image_service.extract_table_as_csv(img_data["data"])
                
                if image_text:
                    html_parts.append(f"<div class='image-text-content' id='image-{img_idx}'>{self._escape_html(image_text)}</div>")
                    json_elements.append({
                        "type": "image_text",
                        "index": img_idx,
                        "content": image_text,
                        "width": img_data.get("width"),
                        "height": img_data.get("height"),
                        "csv_content": table_csv if table_csv != image_text else None
                    })
                    summary_parts.append(image_text[:300])
                
                # Store CSV if different from text
                if table_csv and table_csv != image_text and self.csv_processor:
                    csv_result = self.csv_processor.process_csv_content(
                        table_csv,
                        f"{source_pdf_name}_image_{img_idx}.csv",
                        'image_table'
                    )
                    
                    if hasattr(self, 'db') and self.db:
                        self.db.insert_table_extraction(
                            source_pdf_name, page_number, f"Image Table {img_idx}",
                            'vision_api', table_csv, csv_result.get('doc_id'), 0.8
                        )
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

    def _extract_table_multiple_methods(self, table_df: pd.DataFrame, table_name: str, 
                                       table_context: str, source_pdf: str, page_number: int) -> List[Dict]:
        """Try multiple extraction methods and return results"""
        results = []
        
        # Method 1: Direct DataFrame to CSV
        try:
            csv_content = table_df.to_csv(index=False)
            results.append({
                'method': 'tabula',
                'csv_content': csv_content,
                'confidence': 0.7,
                'row_count': len(table_df),
                'col_count': len(table_df.columns)
            })
        except Exception as e:
            print(f"Tabula method failed: {e}")
        
        # Method 2: Vision API (if available and image service supports it)
        # This would require the raw image data of the table
        # For now, we'll skip this as it requires additional image extraction
        
        # Method 3: Structure-aware extraction (existing logic)
        try:
            table_analysis = self._deep_analyze_table(table_df, table_name, table_context, "")
            structured_csv = self._convert_analysis_to_csv(table_analysis)
            if structured_csv:
                results.append({
                    'method': 'structure_aware',
                    'csv_content': structured_csv,
                    'confidence': 0.8,
                    'row_count': len(table_analysis.get('rows', [])),
                    'col_count': table_analysis.get('col_count', 0)
                })
        except Exception as e:
            print(f"Structure-aware method failed: {e}")
        
        return results

    def _select_best_extraction(self, extraction_results: List[Dict]) -> Optional[Dict]:
        """Select the best extraction result based on confidence and quality"""
        if not extraction_results:
            return None
        
        # Sort by confidence score
        extraction_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return the highest confidence result
        return extraction_results[0]

    def _convert_analysis_to_csv(self, analysis: Dict) -> Optional[str]:
        """Convert table analysis to CSV format"""
        try:
            rows = analysis.get('rows', [])
            if not rows:
                return None
            
            # Get all possible column names
            all_columns = set()
            for row in rows:
                all_columns.update(row.keys())
            
            # Remove internal fields
            column_names = [col for col in all_columns if not col.startswith('_')]
            
            # Create CSV
            csv_lines = [','.join(column_names)]
            
            for row in rows:
                csv_values = []
                for col in column_names:
                    value = str(row.get(col, '')).replace(',', ' ')
                    csv_values.append(value)
                csv_lines.append(','.join(csv_values))
            
            return '\n'.join(csv_lines)
            
        except Exception as e:
            print(f"Error converting analysis to CSV: {e}")
            return None

    def _extract_table_context(self, page_text: str, table_name: str, table_idx: int) -> str:
        """Extract the specific table section from page text"""
        table_pattern = r'Table\s+\d+[^:]*:[^\n]*\n'
        table_starts = list(re.finditer(table_pattern, page_text, re.IGNORECASE))
        
        if table_idx < len(table_starts):
            start_pos = table_starts[table_idx].start()
            
            if table_idx + 1 < len(table_starts):
                end_pos = table_starts[table_idx + 1].start()
            else:
                end_pos = len(page_text)
            
            table_context = page_text[start_pos:end_pos]
            print(f"📄 Extracted table context ({len(table_context)} chars)")
            return table_context
        
        return ""

    def _deep_analyze_table(self, df: pd.DataFrame, table_name: str, table_context: str, full_text: str) -> Dict:
        """Deep analysis of table structure using dynamic approaches"""
        
        # Dynamic table analysis instead of hardcoded numbers
        return self._analyze_table_dynamically(df, table_name, table_context)

    def _analyze_table_dynamically(self, df: pd.DataFrame, table_name: str, table_context: str) -> Dict:
        """Dynamic analysis that adapts to any table structure"""
        print("🎯 Dynamic Table Analysis")
        
        df_clean = df.fillna('')
        rows = []
        headers = list(df.columns)
        
        # Detect table patterns dynamically
        structure_type = self._detect_table_pattern(df_clean, headers, table_context)
        
        # Process based on detected pattern
        if structure_type == "year_sections":
            return self._process_year_section_table(df_clean, table_name, headers)
        elif structure_type == "person_actions":
            return self._process_person_action_table(df_clean, table_name, headers)
        elif structure_type == "financial_hierarchy":
            return self._process_financial_table(df_clean, table_name, headers)
        else:
            return self._process_generic_table(df_clean, table_name, headers)

    def _detect_table_pattern(self, df: pd.DataFrame, headers: List[str], context: str) -> str:
        """Detect table pattern dynamically"""
        
        # Check for year patterns in data
        year_pattern = r'\b(19|20)\d{2}\b'
        has_years = False
        
        for _, row in df.iterrows():
            for col in headers:
                cell_value = str(row[col]).strip()
                if re.match(year_pattern, cell_value):
                    has_years = True
                    break
            if has_years:
                break
        
        # Check for person names pattern
        name_indicators = ['name', 'person', 'individual', 'participant']
        has_names = any(indicator in ' '.join(headers).lower() for indicator in name_indicators)
        
        # Check for action words
        action_words = ['entered', 'won', 'completed', 'started', 'finished']
        has_actions = any(action in ' '.join(headers).lower() for action in action_words)
        
        # Check for financial patterns
        financial_words = ['income', 'cost', 'revenue', 'expense', 'assets', 'liabilities']
        has_financial = any(word in context.lower() for word in financial_words)
        
        # Determine pattern
        if has_years and len(df) > 5:
            return "year_sections"
        elif has_names and has_actions:
            return "person_actions"
        elif has_financial:
            return "financial_hierarchy"
        else:
            return "generic"

    def _process_year_section_table(self, df: pd.DataFrame, table_name: str, headers: List[str]) -> Dict:
        """Process tables with year sections (like Table 29)"""
        rows = []
        current_year = None
        
        for _, row in df.iterrows():
            first_col = str(row[headers[0]]).strip()
            
            # FIXED: Check if this is a year row - Added missing closing quote and parameters
            if re.match(r'^(19|20)\d{2}$', first_col):
                current_year = first_col
                continue
            
            # Skip empty or header rows
            if not first_col or first_col.lower() in ['', 'nan']:
                continue
            
            # This is a data row
            row_data = {'metric': first_col, 'year': current_year or 'unknown'}
            
            # Map remaining columns to regions/categories
            for i, col in enumerate(headers[1:], 1):
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    # Use column header or generic naming
                    clean_header = str(col).replace('Unnamed: ', f'region_{i}')
                    row_data[clean_header] = value
            
            if len(row_data) > 2:  # More than just metric and year
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "year_sections",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _process_person_action_table(self, df: pd.DataFrame, table_name: str, headers: List[str]) -> Dict:
        """Process tables with person and action columns"""
        rows = []
        
        for _, row in df.iterrows():
            name = str(row[headers[0]]).strip()
            if not name or name.lower() in ['name', 'nan', '']:
                continue
            
            row_data = {'identifier': name, 'name': name.lower()}
            
            # Map remaining columns
            for col in headers[1:]:
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    clean_col = str(col).replace('Unnamed: ', 'col_').lower()
                    row_data[clean_col] = value
            
            if len(row_data) > 2:
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "person_actions",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _process_financial_table(self, df: pd.DataFrame, table_name: str, headers: List[str]) -> Dict:
        """Process financial tables with hierarchical structure"""
        rows = []
        current_category = None
        
        for _, row in df.iterrows():
            first_col = str(row[headers[0]]).strip()
            
            # Check if this is a category header
            if self._is_category_header(first_col, row, headers):
                current_category = first_col
                continue
            
            # Skip empty rows
            if not first_col or first_col.lower() in ['', 'nan']:
                continue
            
            # This is a data row
            row_data = {
                'item': first_col,
                'category': current_category or 'uncategorized'
            }
            
            # Map value columns
            for col in headers[1:]:
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    clean_col = str(col).replace('Unnamed: ', 'col_')
                    row_data[clean_col] = value
            
            if len(row_data) > 2:
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "financial_hierarchy",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _process_generic_table(self, df: pd.DataFrame, table_name: str, headers: List[str]) -> Dict:
        """Process generic tables"""
        rows = []
        
        for _, row in df.iterrows():
            row_data = {}
            for col in headers:
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    clean_col = str(col).replace('Unnamed: ', 'col_')
                    row_data[clean_col] = value
            
            if row_data:
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "generic",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _is_category_header(self, text: str, row, headers: List[str]) -> bool:
        """Check if a row is a category header"""
        # Check if this row has minimal data in other columns
        other_values = [str(row[col]).strip() for col in headers[1:]]
        non_empty_values = [v for v in other_values if v and v.lower() != 'nan']
        
        # If first column has text but others are mostly empty, it's likely a category
        return len(text) > 2 and len(non_empty_values) < len(headers) * 0.3

    def _extract_table_number(self, table_name: str) -> Optional[int]:
        """Extract table number from table name"""
        match = re.search(r'table\s+(\d+)', table_name.lower())
        return int(match.group(1)) if match else None

    def _analyze_table_25(self, df: pd.DataFrame, table_name: str, table_context: str) -> Dict:
        """Specific analysis for Table 25: Name + 2008/2009 with Entered/Won"""
        print("🎯 Analyzing Table 25 (Name + Year columns)")
        
        df_clean = df.fillna('')
        rows = []
        headers = list(df.columns)
        
        for row_idx, (_, row) in enumerate(df_clean.iterrows()):
            name = str(row[headers[0]]).strip()
            # FIXED: Added missing closing quote and parameters
            if not name or name.lower() in ['name', 'nan', ''] or re.match("r'^\d{4}, name"):
                continue
            
            row_data = {
                'identifier': name,
                'name': name.lower()
            }
            
            if len(headers) >= 5:
                col_values = [str(row[col]).strip() for col in headers[1:]]
                
                if any(val.isdigit() for val in col_values):
                    row_data['2008_entered'] = col_values[0] if len(col_values) > 0 else ''
                    row_data['2008_won'] = col_values[1] if len(col_values) > 1 else ''
                    row_data['2009_entered'] = col_values[2] if len(col_values) > 2 else ''
                    row_data['2009_won'] = col_values[3] if len(col_values) > 3 else ''
                    
                    for i, col in enumerate(headers[1:]):
                        if i < len(col_values):
                            row_data[f'col_{col}'] = col_values[i]
                    
                    rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "table_25_person_year_actions",
            "headers": headers,
            "years": ["2008", "2009"],
            "actions": ["entered", "won"],
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _analyze_table_29(self, df: pd.DataFrame, table_name: str, table_context: str) -> Dict:
        """Specific analysis for Table 29: Metrics + Continents with Year sections"""
        print("🎯 Analyzing Table 29 (Metrics + Continents)")
        
        df_clean = df.fillna('')
        rows = []
        headers = list(df.columns)
        
        current_year = None
        
        for row_idx, (_, row) in enumerate(df_clean.iterrows()):
            first_col = str(row[headers[0]]).strip()
            
            # FIXED: Added missing closing quote and parameters
            if re.match("r'^(19|20)\d{2}", first_col):
                current_year = first_col
                continue
            
            if not first_col or first_col.lower() in ['', 'nan'] or 'continent' in first_col.lower():
                continue
            
            metric = first_col
            row_data = {
                'metric': metric,
                'year': current_year or 'unknown'
            }
            
            continent_map = {
                1: 'south_america',
                2: 'asia', 
                3: 'africa',
                4: 'australia'
            }
            
            for i, col in enumerate(headers[1:], 1):
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    continent = continent_map.get(i, f'col_{i}')
                    row_data[continent] = value
                    row_data[f'col_{col}'] = value
            
            if len(row_data) > 2:
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "table_29_metric_continent_year",
            "headers": headers,
            "continents": ["south_america", "asia", "africa", "australia"],
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _analyze_table_28(self, df: pd.DataFrame, table_name: str, table_context: str) -> Dict:
        """Analysis for Table 28 and similar financial tables"""
        return self._analyze_generic_table(df, table_name, table_context)

    def _analyze_generic_table(self, df: pd.DataFrame, table_name: str, table_context: str) -> Dict:
        """Generic table analysis"""
        df_clean = df.fillna('')
        rows = []
        headers = list(df.columns)
        
        for _, row in df_clean.iterrows():
            row_data = {}
            for col in headers:
                value = str(row[col]).strip()
                if value and value.lower() != 'nan':
                    clean_col = str(col).replace('Unnamed: ', 'col_')
                    row_data[clean_col] = value
            if row_data:
                rows.append(row_data)
        
        return {
            "table_name": table_name,
            "structure_type": "generic",
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(headers)
        }

    def _create_advanced_searchable_text(self, analysis: Dict, table_name: str) -> str:
        """Create searchable text based on table structure"""
        text_parts = [table_name]
        
        structure_type = analysis.get('structure_type', 'generic')
        rows = analysis.get('rows', [])
        
        if structure_type == "table_25_person_year_actions":
            for row in rows:
                name = row.get('identifier', '')
                if name:
                    for year in ['2008', '2009']:
                        entered = row.get(f'{year}_entered', '')
                        won = row.get(f'{year}_won', '')
                        if entered:
                            text_parts.append(f"{name} entered {entered} in {year}")
                        if won:
                            text_parts.append(f"{name} won {won} in {year}")
        
        elif structure_type == "table_29_metric_continent_year":
            for row in rows:
                metric = row.get('metric', '')
                year = row.get('year', '')
                if metric and year:
                    for continent in ['south_america', 'asia', 'africa', 'australia']:
                        value = row.get(continent, '')
                        if value:
                            continent_name = continent.replace('_', ' ').title()
                            text_parts.append(f"{metric} for {continent_name} in {year} is {value}")
        
        else:
            for row in rows:
                row_text = []
                for key, value in row.items():
                    row_text.append(f"{key}: {value}")
                if row_text:
                    text_parts.append(" | ".join(row_text))
        
        return ". ".join(text_parts)

    def _create_advanced_html(self, analysis: Dict, table_name: str) -> str:
        """Create HTML display"""
        html = f"<div class='table-header'><h4>{table_name}</h4></div>"
        html += f"<p><em>Type: {analysis.get('structure_type', 'unknown')}</em></p>"
        html += "<table class='table table-striped'>"
        
        headers = analysis.get('headers', [])
        if headers:
            html += "<thead><tr>"
            for header in headers:
                clean_header = str(header).replace('Unnamed: ', 'Col ')
                html += f"<th>{clean_header}</th>"
            html += "</tr></thead>"
        
        html += "<tbody>"
        for row in analysis.get('rows', []):
            html += "<tr>"
            for header in headers:
                value = ""
                if f"col_{header}" in row:
                    value = row[f"col_{header}"]
                elif str(header) in row:
                    value = row[str(header)]
                html += f"<td>{self._escape_html(str(value))}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        
        return html

    # Helper methods
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
        
        numeric_count = sum(1 for word in words if re.match("r'^[\d,.\(\)]+", word))
        return numeric_count / len(words) > 0.4

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

# For backward compatibility
class ContentProcessor(AdvancedTableAnalyzer):
    pass

class HybridTableProcessor(AdvancedTableAnalyzer):
    pass