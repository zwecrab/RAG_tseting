import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from rank_bm25 import BM25Okapi
from config import Config

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding_model = None
        self.faiss_index = None
        self.documents = []
        self.bm25 = None
        self.document_embeddings = None
        
    def initialize_models(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
    
    def create_vector_store(self, documents: List[Document]) -> bool:
        try:
            self.initialize_models()
            self.documents = documents
            
            texts = [doc.page_content for doc in documents]
            
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            )
            
            self.document_embeddings = embeddings
            
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings.astype('float32'))
            
            tokenized_corpus = [doc.page_content.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def hybrid_search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        if top_k is None:
            top_k = self.config.RETRIEVER_TOP_K
            
        try:
            self.initialize_models()
            
            vector_results = self._vector_search(query, top_k * 2)
            bm25_results = self._bm25_search(query, top_k * 2)
            
            combined_results = self._combine_results(vector_results, bm25_results, top_k)
            
            return combined_results
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents) and scores[idx] > 0:
                results.append((self.documents[idx], float(scores[idx])))
        
        return results
    
    def _combine_results(self, vector_results: List[Tuple[Document, float]], 
                        bm25_results: List[Tuple[Document, float]], 
                        top_k: int) -> List[Tuple[Document, float]]:
        
        vector_scores = {id(doc): score for doc, score in vector_results}
        bm25_scores = {id(doc): score for doc, score in bm25_results}
        
        all_docs = set()
        all_docs.update(id(doc) for doc, _ in vector_results)
        all_docs.update(id(doc) for doc, _ in bm25_results)
        
        doc_mapping = {}
        for doc, _ in vector_results + bm25_results:
            doc_mapping[id(doc)] = doc
        
        combined_scores = []
        for doc_id in all_docs:
            vector_score = vector_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            
            if bm25_score > 0:
                bm25_score = bm25_score / 10.0
            
            combined_score = (self.config.VECTOR_WEIGHT * vector_score + 
                             self.config.BM25_WEIGHT * bm25_score)
            
            combined_scores.append((doc_mapping[doc_id], combined_score))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return combined_scores[:top_k]
    
    def save_vector_store(self, filepath: str) -> bool:
        try:
            store_data = {
                'documents': self.documents,
                'document_embeddings': self.document_embeddings,
                'bm25_corpus': [doc.page_content.split() for doc in self.documents]
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(store_data, f)
            
            faiss_path = filepath.replace('.pkl', '.faiss')
            faiss.write_index(self.faiss_index, faiss_path)
            
            return True
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
            return False
    
    def load_vector_store(self, filepath: str) -> bool:
        try:
            self.initialize_models()
            
            with open(filepath, 'rb') as f:
                store_data = pickle.load(f)
            
            self.documents = store_data['documents']
            self.document_embeddings = store_data['document_embeddings']
            
            faiss_path = filepath.replace('.pkl', '.faiss')
            if os.path.exists(faiss_path):
                self.faiss_index = faiss.read_index(faiss_path)
            
            self.bm25 = BM25Okapi(store_data['bm25_corpus'])
            
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def get_similar_documents(self, query: str, doc_type: str = None, page_num: int = None) -> List[Document]:
        results = self.hybrid_search(query)
        
        filtered_results = []
        for doc, score in results:
            if doc_type and doc.metadata.get('type') != doc_type:
                continue
            if page_num and doc.metadata.get('page') != page_num:
                continue
            filtered_results.append(doc)
        
        return filtered_results
