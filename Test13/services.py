import base64
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import List, Dict
import json

class TableExtractor:
    def __init__(self, api_key: str, azure_endpoint: str = None, deployment_name: str = "gpt-4o", api_version: str = "2024-02-15-preview"):
        """Initialize for Azure OpenAI Service or standard OpenAI"""
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        if azure_endpoint:
            # Azure format
            self.base_url = f"{azure_endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        else:
            # Standard OpenAI format
            self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def encode_image(self, image_bytes: bytes) -> str:
        """Encode image to base64 for API"""
        try:
            if isinstance(image_bytes, bytes):
                return base64.b64encode(image_bytes).decode('utf-8')
            else:
                with open(image_bytes, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise Exception(f"Error encoding image: {str(e)}")
    
    def create_extraction_prompt(self) -> str:
        """Smart prompt that handles different table types correctly"""
        return """
Convert this table to CSV format. Analyze the structure first.

TABLE TYPES:

TYPE 1 - Simple row/column table:
Row headers are data labels, column headers are categories/years
Example: Subjects in rows, Years in columns
Output: Subject,2006,2007,2008,2009

TYPE 2 - Complex multi-level headers:
Headers span multiple rows/columns  
Example: Regions with sub-years
Output: Metric,South America 2010,Asia 2010,etc.

RULES:
1. For simple tables: Keep row headers as first column, years as separate columns
2. For complex tables: Flatten multi-level headers only
3. Remove commas from numbers: "1,234" → "1234"  
4. Keep text commas as spaces: "A, B" → "A B"

EXAMPLES:

Simple table (Type 1):
| Subject | 2006 | 2007 |
|---------|------|------|
| Math    | A,B  | B,C  |
| English | A    | B    |

Output:
Subject,2006,2007
Math,A B,B C
English,A,B

Complex table (Type 2):
| Region | South America | Asia |
|        | 2010 | 2009   | 2010 |

Output:
Region,South America 2010,South America 2009,Asia 2010

Convert the table using the correct type:
"""
    
    def extract_table_to_csv(self, image_data) -> str:
        """Main extraction function"""
        print(f"Processing table image...")
        
        # Encode image
        try:
            base64_image = self.encode_image(image_data)
        except Exception as e:
            return f"Error: {str(e)}"
        
        # Prepare API request
        if self.azure_endpoint:
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        
        payload = {
            "model": self.deployment_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.create_extraction_prompt()
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        # Make API call
        try:
            print("Calling Vision API...")
            response = requests.post(self.base_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return f"API Error: {response.status_code}"
            
            result = response.json()
            csv_content = result['choices'][0]['message']['content'].strip()
            
            print("✅ Table extraction successful!")
            return csv_content
            
        except Exception as e:
            return f"Extraction error: {str(e)}"

class ImageToTextService:
    def __init__(self, api_key: str, azure_endpoint: str = None, deployment_name: str = "gpt-4o"):
        if not api_key:
            raise ValueError("API key is required.")
        
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        
        # Initialize table extractor
        self.table_extractor = TableExtractor(api_key, azure_endpoint, deployment_name)
        
        if azure_endpoint:
            self.headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            self.api_url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
        else:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            self.api_url = "https://api.openai.com/v1/chat/completions"

    def get_text_from_image(self, image_bytes: bytes) -> str:
        """Extract text from image"""
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "model": self.deployment_name,
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
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            return content
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return ""

    def extract_table_as_csv(self, image_bytes: bytes) -> str:
        """Extract table as CSV format"""
        return self.table_extractor.extract_table_to_csv(image_bytes)

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        print(f"✅ Embedding service loaded with model: {model_name}")

    def create_embeddings(self, text_chunks: List[str]):
        """Create embeddings for text chunks"""
        if not text_chunks:
            return []
        return self.model.encode(text_chunks, show_progress_bar=len(text_chunks) > 10)

    def create_single_embedding(self, text: str):
        """Create single embedding"""
        return self.model.encode([text])[0]

class LLMService:
    def __init__(self, api_key: str, azure_endpoint: str = None, deployment_name: str = "gpt-4o"):
        """Initialize LLM service with GPT-4o for better answer generation"""
        self.api_key = api_key
        self.azure_endpoint = azure_endpoint
        self.deployment_name = deployment_name
        
        if azure_endpoint:
            self.headers = {
                "Content-Type": "application/json",
                "api-key": api_key
            }
            self.api_url = f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2024-02-15-preview"
        else:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            self.api_url = "https://api.openai.com/v1/chat/completions"
        
        print(f"✅ LLM service initialized with {deployment_name} via {'Azure OpenAI' if azure_endpoint else 'OpenAI'}")

    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using GPT-4o"""
        prompt = f"""You are an expert assistant analyzing document content and data. Based on the provided context, answer the user's question accurately and concisely.

Context:
---
{context}
---

Instructions:
- Use ONLY the information provided in the context
- If the answer is not in the context, say "I cannot find that information in the provided context"
- Be precise and specific in your answer
- If providing numerical data, include the exact values from the context
- Keep your response concise but complete

Question: {query}

Answer:"""

        payload = {
            "model": self.deployment_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context. Be accurate, concise, and only use information from the given context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,  # Low temperature for factual responses
            "top_p": 0.9
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            return answer
            
        except requests.exceptions.RequestException as e:
            print(f"❌ LLM API request failed: {e}")
            return "I apologize, but I'm unable to generate an answer due to a technical issue."
        except KeyError as e:
            print(f"❌ LLM response parsing failed: {e}")
            return "I apologize, but I received an unexpected response format."
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return "Could not generate an answer due to an unexpected error."
    
    def generate_structured_answer(self, query: str, context: str, format_type: str = "json") -> str:
        """Generate structured answer in specific format"""
        format_instructions = {
            "json": "Provide your answer in JSON format with 'answer' and 'confidence' fields.",
            "csv": "If the answer contains tabular data, format it as CSV.",
            "list": "If the answer contains multiple items, format them as a numbered list.",
            "summary": "Provide a brief summary followed by key details."
        }
        
        instruction = format_instructions.get(format_type, "Provide a clear, well-structured answer.")
        
        prompt = f"""Based on the provided context, answer the user's question. {instruction}

Context:
---
{context}
---

Question: {query}

Answer:"""

        payload = {
            "model": self.deployment_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 700,
            "temperature": 0.1
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            print(f"❌ Structured answer generation failed: {e}")
            return self.generate_answer(query, context)  # Fallback to regular answer

class CSVQueryProcessor:
    def __init__(self, db, embedding_service):
        self.db = db
        self.embedding_service = embedding_service
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Semantic search across CSV data"""
        query_embedding = self.embedding_service.create_single_embedding(query)
        return self.db.search_csv_data(query_embedding, k)
    
    def exact_value_search(self, column: str, value: str) -> List[Dict]:
        """Search for exact values in specific columns"""
        return self.db.search_by_column_value(column, value)
    
    def hybrid_search(self, query: str, column_filter: str = None) -> List[Dict]:
        """Combine semantic and exact matching"""
        results = self.semantic_search(query)
        
        if column_filter:
            filtered_results = []
            for result in results:
                if column_filter in result.get('column_names', []):
                    filtered_results.append(result)
            return filtered_results
        
        return results