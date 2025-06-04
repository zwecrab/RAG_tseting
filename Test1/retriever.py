from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, documents: List[Document], embeddings):
        self.documents = documents
        self.embeddings = embeddings
        
        self.vector_store = FAISS.from_documents(documents, embeddings)
        
        corpus = [doc.page_content for doc in documents]
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        vector_results = self.vector_store.similarity_search(query, k=k)
        
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [self.documents[i] for i in bm25_indices]
        
        combined_results = {}
        for doc in vector_results:
            doc_key = doc.page_content[:100]
            combined_results[doc_key] = doc
        
        for doc in bm25_results:
            doc_key = doc.page_content[:100]
            if doc_key not in combined_results:
                combined_results[doc_key] = doc
        
        return list(combined_results.values())[:k]