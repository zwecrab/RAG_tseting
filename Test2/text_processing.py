import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

class TextProcessor:
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
    
    def create_documents(self, extracted_content: List[Dict[str, Any]], filename: str) -> List[Document]:
        documents = []
        
        for item in extracted_content:
            if item['type'] == 'text':
                text_docs = self._process_text_content(item, filename)
                documents.extend(text_docs)
            elif item['type'] == 'table':
                table_doc = self._process_table_content(item, filename)
                if table_doc:
                    documents.append(table_doc)
            elif item['type'] == 'image':
                image_doc = self._process_image_content(item, filename)
                if image_doc:
                    documents.append(image_doc)
        
        return documents
    
    def _process_text_content(self, item: Dict[str, Any], filename: str) -> List[Document]:
        cleaned_text = self._clean_text(item['content'])
        chunks = self.text_splitter.split_text(cleaned_text)
        
        documents = []
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue
                
            doc = Document(
                page_content=chunk,
                metadata={
                    'source': filename,
                    'page': item['page'],
                    'type': 'text',
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'extractor': item.get('source', 'unknown')
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_table_content(self, item: Dict[str, Any], filename: str) -> Document:
        return Document(
            page_content=item['content'],
            metadata={
                'source': filename,
                'page': item['page'],
                'type': 'table',
                'table_index': item['metadata']['table_index'],
                'rows': item['metadata']['rows'],
                'columns': item['metadata']['columns']
            }
        )
    
    def _process_image_content(self, item: Dict[str, Any], filename: str) -> Document:
        if not item['metadata']['has_text']:
            return None
            
        return Document(
            page_content=f"Image content from page {item['page']}: {item['content']}",
            metadata={
                'source': filename,
                'page': item['page'],
                'type': 'image',
                'image_index': item['metadata']['image_index'],
                'image_path': item.get('image_path', ''),
                'has_ocr_text': item['metadata']['has_text']
            }
        )
    
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = text.strip()
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'throughout', 'inside',
            'outside', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
            'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now', 'also'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))
