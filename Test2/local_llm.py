import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from typing import List, Dict, Any
from langchain.schema import Document

class LocalLLM:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def initialize_model(self):
        if self.model is None:
            try:
                if "flan-t5" in self.model_name.lower():
                    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                        device_map="auto" if self.device == "cuda" else None
                    )
                
                self.qa_pipeline = pipeline(
                    "text2text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
                
                print(f"Model {self.model_name} loaded successfully on {self.device}")
                
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self._fallback_to_simple_model()
    
    def _fallback_to_simple_model(self):
        try:
            self.model_name = "google/flan-t5-small"
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
                max_length=256
            )
            
            print(f"Fallback to {self.model_name}")
            
        except Exception as e:
            print(f"Error loading fallback model: {str(e)}")
            self.qa_pipeline = None
    
    def generate_answer(self, question: str, context_documents: List[Document]) -> Dict[str, Any]:
        if not self.qa_pipeline:
            self.initialize_model()
        
        if not self.qa_pipeline:
            return {
                "answer": "Error: Could not load language model",
                "confidence": 0.0,
                "source_documents": []
            }
        
        try:
            context = self._prepare_context(context_documents)
            prompt = self._create_prompt(question, context)
            
            response = self.qa_pipeline(
                prompt,
                max_length=400,
                min_length=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            answer = response[0]['generated_text'].strip()
            confidence = self._calculate_confidence(answer, context_documents)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "source_documents": context_documents[:3]
            }
            
        except Exception as e:
            print(f"Error generating answer: {str(e)}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "confidence": 0.0,
                "source_documents": context_documents[:3] if context_documents else []
            }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        context_parts = []
        max_context_length = 1500
        current_length = 0
        
        for doc in documents:
            doc_text = doc.page_content
            doc_info = f"From {doc.metadata.get('source', 'unknown')} (Page {doc.metadata.get('page', 'unknown')})"
            
            if doc.metadata.get('type') == 'table':
                doc_text = f"[TABLE] {doc_text}"
            elif doc.metadata.get('type') == 'image':
                doc_text = f"[IMAGE TEXT] {doc_text}"
            
            full_text = f"{doc_info}: {doc_text}"
            
            if current_length + len(full_text) > max_context_length:
                break
                
            context_parts.append(full_text)
            current_length += len(full_text)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        prompt = f"""Context information:
{context}

Question: {question}

Based on the context information provided above, please provide a comprehensive and accurate answer to the question. If the information is not sufficient to answer the question completely, please indicate what information is missing.

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, answer: str, documents: List[Document]) -> float:
        if not answer or len(answer) < 10:
            return 0.1
        
        if "error" in answer.lower() or "cannot" in answer.lower():
            return 0.3
        
        if not documents:
            return 0.4
        
        answer_words = set(answer.lower().split())
        context_words = set()
        for doc in documents:
            context_words.update(doc.page_content.lower().split())
        
        overlap = len(answer_words.intersection(context_words))
        total_answer_words = len(answer_words)
        
        if total_answer_words == 0:
            return 0.5
        
        overlap_ratio = overlap / total_answer_words
        base_confidence = min(0.9, 0.5 + overlap_ratio * 0.4)
        
        return base_confidence
    
    def summarize_document(self, documents: List[Document]) -> str:
        if not self.qa_pipeline:
            self.initialize_model()
        
        if not documents:
            return "No documents to summarize"
        
        try:
            context = self._prepare_context(documents)
            prompt = f"""Please provide a comprehensive summary of the following content:

{context}

Summary:"""
            
            response = self.qa_pipeline(
                prompt,
                max_length=300,
                min_length=100,
                do_sample=True,
                temperature=0.5
            )
            
            return response[0]['generated_text'].strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"