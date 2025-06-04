from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class SmartChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
    
    def chunk_content(self, content_list: List[Dict[str, Any]]) -> List[Document]:
        documents = []
        
        for item in content_list:
            if item['type'] == 'text':
                chunks = self.text_splitter.split_text(item['content'])
                for chunk_idx, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            **item['metadata'],
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks)
                        }
                    )
                    documents.append(doc)
            
            elif item['type'] == 'table':
                doc = Document(
                    page_content=f"Table from page {item['page']}:\n{item['content']}",
                    metadata=item['metadata']
                )
                documents.append(doc)
        
        return documents