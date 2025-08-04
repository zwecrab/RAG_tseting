import pymupdf
import tabula
import pandas as pd
import os
from typing import List, Dict, Any, Optional

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

            # Enhanced table extraction with fallback
            try:
                tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True, lattice=True)
                if tables:
                    for table_df in tables:
                        if not table_df.empty:
                            page_content["tables"].append(table_df)
            except Exception as e:
                print(f"Warning: Tabula failed on page {page_num + 1}. Error: {e}")
                
                # Try alternative tabula settings
                try:
                    tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True, stream=True)
                    if tables:
                        for table_df in tables:
                            if not table_df.empty:
                                page_content["tables"].append(table_df)
                except Exception as e2:
                    print(f"Warning: Alternative tabula method also failed: {e2}")

            # Extract images with enhanced metadata
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
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
                            "bbox": bbox,
                            "xref": xref
                        })
                    pix = None
                except Exception as e:
                    print(f"Warning: Could not extract image {img_index} from page {page_num + 1}: {e}")

            raw_content.append(page_content)
        
        doc.close()
        print(f"✅ Extracted content from {len(raw_content)} pages.")
        return raw_content

class CSVExtractor:
    """New extractor for CSV files"""
    
    def __init__(self):
        self.supported_extensions = ['.csv', '.tsv', '.txt']
    
    def is_supported_file(self, filepath: str) -> bool:
        """Check if file is a supported CSV format"""
        return any(filepath.lower().endswith(ext) for ext in self.supported_extensions)
    
    def extract_content(self, filepath: str) -> Dict[str, Any]:
        """Extract content from CSV file"""
        if not self.is_supported_file(filepath):
            raise ValueError(f"Unsupported file type. Supported: {self.supported_extensions}")
        
        try:
            filename = os.path.basename(filepath)
            print(f"Extracting CSV content from: {filename}")
            
            # Try to read CSV with multiple methods
            df = self._read_csv_flexible(filepath)
            
            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "Could not read CSV file or file is empty",
                    "filepath": filepath
                }
            
            # Analyze CSV structure
            analysis = self._analyze_csv_structure(df, filename)
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "dataframe": df,
                "analysis": analysis,
                "raw_content": df.to_dict('records')
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "filepath": filepath
            }
    
    def _read_csv_flexible(self, filepath: str) -> Optional[pd.DataFrame]:
        """Try multiple methods to read CSV file"""
        methods = [
            # Standard CSV
            lambda: pd.read_csv(filepath),
            
            # Different encodings
            lambda: pd.read_csv(filepath, encoding='latin1'),
            lambda: pd.read_csv(filepath, encoding='cp1252'),
            lambda: pd.read_csv(filepath, encoding='iso-8859-1'),
            
            # Different separators
            lambda: pd.read_csv(filepath, sep=';'),
            lambda: pd.read_csv(filepath, sep='\t'),
            lambda: pd.read_csv(filepath, sep='|'),
            
            # Handle headers
            lambda: pd.read_csv(filepath, header=None),
            lambda: pd.read_csv(filepath, header=1),
            
            # Handle quotes
            lambda: pd.read_csv(filepath, quotechar='"'),
            lambda: pd.read_csv(filepath, quotechar="'"),
            
            # Error handling
            lambda: pd.read_csv(filepath, on_bad_lines='skip'),
            lambda: pd.read_csv(filepath, encoding='utf-8', errors='replace'),
        ]
        
        for i, method in enumerate(methods):
            try:
                df = method()
                if df is not None and not df.empty:
                    print(f"✅ Successfully read CSV using method {i+1}")
                    return df
            except Exception as e:
                continue
        
        print("❌ All CSV reading methods failed")
        return None
    
    def _analyze_csv_structure(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Analyze CSV structure and quality"""
        analysis = {
            "filename": filename,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "sample_data": df.head().to_dict('records'),
            "quality_metrics": {}
        }
        
        # Quality metrics
        total_cells = df.shape[0] * df.shape[1]
        null_cells = df.isnull().sum().sum()
        
        analysis["quality_metrics"] = {
            "total_cells": total_cells,
            "null_cells": null_cells,
            "completeness": (total_cells - null_cells) / total_cells if total_cells > 0 else 0,
            "duplicate_rows": df.duplicated().sum(),
            "empty_columns": (df.isnull().all()).sum()
        }
        
        # Column analysis
        column_analysis = {}
        for col in df.columns:
            unique_vals = df[col].nunique()
            total_vals = len(df[col].dropna())
            
            column_analysis[col] = {
                "unique_values": unique_vals,
                "uniqueness_ratio": unique_vals / total_vals if total_vals > 0 else 0,
                "most_common": df[col].value_counts().head(3).to_dict(),
                "data_type": self._infer_column_type(df[col])
            }
        
        analysis["column_analysis"] = column_analysis
        
        return analysis
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Infer the semantic type of a column with better error handling"""
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'empty'
        
        # Check numeric
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        
        # Check datetime (with warning suppression)
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(non_null_series, errors='raise')
            return 'datetime'
        except:
            pass
        
        # Check boolean
        if series.dtype == bool or set(non_null_series.unique().astype(str)) <= {'True', 'False', '1', '0', 'yes', 'no'}:
            return 'boolean'
        
        # Check categorical
        unique_ratio = len(non_null_series.unique()) / len(non_null_series)
        if unique_ratio < 0.5 and len(non_null_series.unique()) < 50:
            return 'categorical'
        
        return 'text'

class UniversalExtractor:
    """Universal extractor that handles both PDF and CSV files"""
    
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.csv_extractor = CSVExtractor()
    
    def extract_content(self, filepath: str) -> Dict[str, Any]:
        """Extract content from file based on extension"""
        if not os.path.exists(filepath):
            return {
                "success": False,
                "error": f"File not found: {filepath}",
                "file_type": "unknown"
            }
        
        filename = os.path.basename(filepath)
        file_ext = os.path.splitext(filename)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return {
                    "success": True,
                    "file_type": "pdf",
                    "filepath": filepath,
                    "filename": filename,
                    "raw_content": self.pdf_extractor.extract_content(filepath)
                }
            
            elif self.csv_extractor.is_supported_file(filepath):
                result = self.csv_extractor.extract_content(filepath)
                result["file_type"] = "csv"
                return result
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}",
                    "file_type": "unsupported",
                    "filepath": filepath
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_type": "error",
                "filepath": filepath
            }
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return ['.pdf'] + self.csv_extractor.supported_extensions
    
    def is_supported_file(self, filepath: str) -> bool:
        """Check if file is supported"""
        file_ext = os.path.splitext(filepath)[1].lower()
        return file_ext in self.get_supported_extensions()

class BatchExtractor:
    """Batch processing for multiple files"""
    
    def __init__(self):
        self.extractor = UniversalExtractor()
    
    def extract_from_directory(self, directory_path: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """Extract content from all supported files in directory"""
        results = []
        
        if not os.path.exists(directory_path):
            return [{"success": False, "error": f"Directory not found: {directory_path}"}]
        
        # Get all files
        files = []
        if recursive:
            for root, dirs, filenames in os.walk(directory_path):
                for filename in filenames:
                    files.append(os.path.join(root, filename))
        else:
            files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                    if os.path.isfile(os.path.join(directory_path, f))]
        
        # Filter supported files
        supported_files = [f for f in files if self.extractor.is_supported_file(f)]
        
        print(f"Found {len(supported_files)} supported files in {directory_path}")
        
        for filepath in supported_files:
            print(f"Processing: {os.path.basename(filepath)}")
            result = self.extractor.extract_content(filepath)
            results.append(result)
        
        return results
    
    def extract_from_file_list(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Extract content from list of file paths"""
        results = []
        
        for filepath in file_paths:
            if self.extractor.is_supported_file(filepath):
                print(f"Processing: {os.path.basename(filepath)}")
                result = self.extractor.extract_content(filepath)
                results.append(result)
            else:
                results.append({
                    "success": False,
                    "error": f"Unsupported file type: {filepath}",
                    "filepath": filepath
                })
        
        return results

# For backward compatibility
class ContentExtractor(PDFExtractor):
    pass