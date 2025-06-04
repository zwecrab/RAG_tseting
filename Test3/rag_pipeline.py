from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load models once at module level
encoder = SentenceTransformer('all-MiniLM-L6-v2')
generator = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)

def build_index(texts):
    """Build a FAISS index from a list of text documents."""
    embeddings = encoder.encode(texts, show_progress_bar=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

def retrieve_passages(index, texts, query, k=3):
    """Retrieve the top-k relevant passages for a query."""
    query_embedding = encoder.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [texts[i] for i in indices[0]]

def generate_answer(question, passages):
    """Generate an answer using the retrieved passages."""
    input_text = f"question: {question} context: {' '.join(passages)}"
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids.to(device)
    output_ids = generator.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer