# services.py

import base64
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict

class ImageToTextService:
    """
    Uses OpenAI's GPT-4o model to extract text from images.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_text_from_image(self, image_bytes: bytes) -> str:
        """
        Sends an image to the GPT-4o API and returns the extracted text.
        """
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text from this image. If it is a table, try to preserve the row and column structure. Only return the extracted text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            content = response.json()['choices'][0]['message']['content']
            return content
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return "" # Return empty string on failure

class EmbeddingService:
    """
    Handles the creation of text embeddings using a SentenceTransformer model.
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding service loaded with model: {model_name}")

    def create_embeddings(self, text_chunks: List[str]):
        """
        Converts a list of text chunks into a list of vector embeddings.
        """
        return self.model.encode(text_chunks, show_progress_bar=True)

class LLMService:
    """
    Handles generating final, human-readable answers using a generative LLM.
    """
    def __init__(self, model_name: str):
        self.generator = pipeline("text2text-generation", model=model_name)
        print(f"✅ LLM service loaded with model: {model_name}")

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates a final answer based on the retrieved context and user query.
        """
        prompt = f"""
        Context:
        ---
        {context}
        ---
        Based only on the context provided, answer the following question concisely.
        If the information is not in the context, say so.

        Question: {query}

        Answer:
        """
        try:
            response = self.generator(prompt, max_length=150, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return "Could not generate an answer."
