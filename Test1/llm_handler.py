from typing import List
from langchain.schema import Document
import re

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LocalLLMHandler:
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.device = "cuda" if TRANSFORMERS_AVAILABLE and torch and torch.cuda.is_available() else "cpu"
        self.use_simple_extraction = True
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model=model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.use_simple_extraction = False
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                print("Falling back to simple text extraction")
    
    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_docs[:3]])
        
        if not self.use_simple_extraction and TRANSFORMERS_AVAILABLE:
            try:
                result = self.qa_pipeline(question=question, context=context)
                return result['answer']
            except Exception as e:
                print(f"Error with transformer model: {e}")
                return self._simple_answer_extraction(question, context)
        else:
            return self._simple_answer_extraction(question, context)
    
    def _simple_answer_extraction(self, question: str, context: str) -> str:
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        
        # Score sentences based on question keywords
        question_words = set(re.findall(r'\w+', question_lower))
        question_words.discard('what')
        question_words.discard('is')
        question_words.discard('are')
        question_words.discard('how')
        question_words.discard('why')
        question_words.discard('when')
        question_words.discard('where')
        
        best_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r'\w+', sentence_lower))
            
            # Calculate overlap score
            overlap = len(question_words.intersection(sentence_words))
            if overlap > 0:
                best_sentences.append((sentence, overlap))
        
        # Sort by overlap score and take top sentences
        best_sentences.sort(key=lambda x: x[1], reverse=True)
        
        if best_sentences:
            # Return the best matching sentence(s)
            if len(best_sentences) >= 2 and best_sentences[0][1] == best_sentences[1][1]:
                return f"{best_sentences[0][0].strip()}. {best_sentences[1][0].strip()}."
            else:
                return best_sentences[0][0].strip() + "."
        
        # If no good matches, return first meaningful sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                return sentence + "."
        
        return "I couldn't find a specific answer in the provided context. Please try rephrasing your question or check if the information exists in the uploaded documents."