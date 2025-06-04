from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import json
from datetime import datetime

from pdf_processor import PDFProcessor
from text_processing import TextProcessor
from vector_store import VectorStore
from local_llm import LocalLLM
from config import Config

class QASystem:
    def __init__(self):
        self.config = Config()
        self.config.create_directories()
        
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.llm = LocalLLM()
        
        self.loaded_documents = {}
        self.document_metadata = {}
        
    def process_pdf(self, pdf_path: str, filename: str) -> Dict[str, Any]:
        try:
            print(f"Processing PDF: {filename}")
            
            extracted_content = self.pdf_processor.extract_content_from_pdf(pdf_path)
            if not extracted_content:
                return {"status": "error", "message": "No content extracted from PDF"}
            
            documents = self.text_processor.create_documents(extracted_content, filename)
            if not documents:
                return {"status": "error", "message": "No documents created from extracted content"}
            
            success = self.vector_store.create_vector_store(documents)
            if not success:
                return {"status": "error", "message": "Failed to create vector store"}
            
            self.loaded_documents[filename] = documents
            self.document_metadata[filename] = {
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
                "total_chunks": len(documents),
                "content_types": self._get_content_types(documents),
                "total_pages": max([doc.metadata.get('page', 1) for doc in documents]),
                "file_hash": self._calculate_file_hash(pdf_path)
            }
            
            self._save_document_metadata(filename)
            
            return {
                "status": "success",
                "message": f"Successfully processed {filename}",
                "chunks": len(documents),
                "content_types": self.document_metadata[filename]["content_types"]
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Error processing PDF: {str(e)}"}
    
    def ask_question(self, question: str, selected_files: List[str] = None, 
                    content_type: str = None, page_num: int = None) -> Dict[str, Any]:
        try:
            if not self.loaded_documents:
                return {
                    "answer": "No documents loaded. Please upload and process PDF files first.",
                    "confidence": 0.0,
                    "source_documents": []
                }
            
            all_documents = []
            if selected_files:
                for filename in selected_files:
                    if filename in self.loaded_documents:
                        all_documents.extend(self.loaded_documents[filename])
            else:
                for docs in self.loaded_documents.values():
                    all_documents.extend(docs)
            
            if not all_documents:
                return {
                    "answer": "No documents found for the selected files.",
                    "confidence": 0.0,
                    "source_documents": []
                }
            
            temp_vector_store = VectorStore()
            temp_vector_store.create_vector_store(all_documents)
            
            relevant_docs = temp_vector_store.get_similar_documents(
                question, doc_type=content_type, page_num=page_num
            )
            
            if not relevant_docs:
                return {
                    "answer": "No relevant information found for your question.",
                    "confidence": 0.0,
                    "source_documents": []
                }
            
            result = self.llm.generate_answer(question, relevant_docs)
            
            return result
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "source_documents": []
            }
    
    def get_document_summary(self, filename: str) -> str:
        if filename not in self.loaded_documents:
            return "Document not found"
        
        documents = self.loaded_documents[filename][:10]
        return self.llm.summarize_document(documents)
    
    def search_documents(self, query: str, selected_files: List[str] = None) -> List[Dict[str, Any]]:
        try:
            all_documents = []
            if selected_files:
                for filename in selected_files:
                    if filename in self.loaded_documents:
                        all_documents.extend(self.loaded_documents[filename])
            else:
                for docs in self.loaded_documents.values():
                    all_documents.extend(docs)
            
            if not all_documents:
                return []
            
            temp_vector_store = VectorStore()
            temp_vector_store.create_vector_store(all_documents)
            
            results = temp_vector_store.hybrid_search(query, top_k=10)
            
            search_results = []
            for doc, score in results:
                search_results.append({
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "score": score,
                    "metadata": doc.metadata,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "page": doc.metadata.get('page', 'Unknown'),
                    "type": doc.metadata.get('type', 'text')
                })
            
            return search_results
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        if not self.document_metadata:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "content_types": {},
                "documents": []
            }
        
        total_chunks = sum(meta["total_chunks"] for meta in self.document_metadata.values())
        all_content_types = {}
        
        for meta in self.document_metadata.values():
            for content_type, count in meta["content_types"].items():
                all_content_types[content_type] = all_content_types.get(content_type, 0) + count
        
        return {
            "total_documents": len(self.document_metadata),
            "total_chunks": total_chunks,
            "content_types": all_content_types,
            "documents": list(self.document_metadata.values())
        }
    
    def remove_document(self, filename: str) -> bool:
        try:
            if filename in self.loaded_documents:
                del self.loaded_documents[filename]
            
            if filename in self.document_metadata:
                del self.document_metadata[filename]
            
            metadata_file = self.config.VECTOR_STORE_DIR / f"{filename}_metadata.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"Error removing document: {str(e)}")
            return False
    
    def get_document_content_by_page(self, filename: str, page_num: int) -> List[Dict[str, Any]]:
        if filename not in self.loaded_documents:
            return []
        
        page_content = []
        for doc in self.loaded_documents[filename]:
            if doc.metadata.get('page') == page_num:
                page_content.append({
                    "content": doc.page_content,
                    "type": doc.metadata.get('type', 'text'),
                    "metadata": doc.metadata
                })
        
        return page_content
    
    def _get_content_types(self, documents) -> Dict[str, int]:
        content_types = {}
        for doc in documents:
            doc_type = doc.metadata.get('type', 'text')
            content_types[doc_type] = content_types.get(doc_type, 0) + 1
        return content_types
    
    def _calculate_file_hash(self, filepath: str) -> str:
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _save_document_metadata(self, filename: str):
        metadata_file = self.config.VECTOR_STORE_DIR / f"{filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.document_metadata[filename], f, indent=2)
    
    def export_qa_history(self) -> Dict[str, Any]:
        return {
            "export_date": datetime.now().isoformat(),
            "documents": self.document_metadata,
            "system_config": {
                "chunk_size": self.config.CHUNK_SIZE,
                "chunk_overlap": self.config.CHUNK_OVERLAP,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "retriever_top_k": self.config.RETRIEVER_TOP_K
            }
        }
