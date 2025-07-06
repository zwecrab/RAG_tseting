# main.py

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pdf_processor import extract_text_from_pdf
import pickle

# --- Configuration ---
PDF_FOLDER = "pdfs"
INDEX_FILE = "faiss_index.bin"
DATA_FILE = "text_data.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 1. Indexing Function ---
def build_index():
    if not os.path.exists(PDF_FOLDER):
        print(f"Error: Folder '{PDF_FOLDER}' not found.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in '{PDF_FOLDER}'.")
        return

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    all_chunks_with_metadata = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        
        # === THIS IS THE MAIN CHANGE ===
        # The processor now returns a clean list of small chunks.
        chunks = extract_text_from_pdf(pdf_path)
        
        for chunk_text in chunks:
            all_chunks_with_metadata.append({'source': pdf_file, 'text': chunk_text})
        # ===============================

    if not all_chunks_with_metadata:
        print("No text could be extracted. Halting index build.")
        return

    print(f"\nGenerating embeddings for {len(all_chunks_with_metadata)} text chunks...")
    
    # Extract just the text for embedding
    texts_to_embed = [item['text'] for item in all_chunks_with_metadata]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))

    print(f"Index built successfully with {index.ntotal} vectors.")

    faiss.write_index(index, INDEX_FILE)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(all_chunks_with_metadata, f)
    
    print(f"Index saved to {INDEX_FILE}")
    print(f"Text data saved to {DATA_FILE}")

# --- 2. Searching Function (No changes needed here) ---
def search(query_text, k=5):
    if not os.path.exists(INDEX_FILE):
        print("Error: Index not found. Please run 'index' first.")
        return

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    with open(DATA_FILE, 'rb') as f:
        chunk_metadata = pickle.load(f)

    print(f"Searching for: '{query_text}'")
    query_vector = model.encode([query_text])
    
    distances, indices = index.search(np.array(query_vector), k)

    print("\n--- Search Results ---")
    for i, idx in enumerate(indices[0]):
        if idx < len(chunk_metadata):
            metadata = chunk_metadata[idx]
            # Convert FAISS L2 distance to a pseudo-similarity score (0-1)
            similarity = 1 / (1 + distances[0][i]) 
            print(f"\n{i+1}. (Source: {metadata['source']}, Similarity: {similarity:.4f})")
            print("-" * 20)
            print(metadata['text'])
    print("\n--- End of Results ---")

# --- 3. Main CLI (No changes needed here) ---
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        print("Commands:")
        print("  index      - Process PDFs and create an index.")
        print("  search     - Search with a query. e.g., python main.py search \"your query\"")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == 'index':
        build_index()
    elif command == 'search':
        if len(sys.argv) < 3:
            print("Usage: python main.py search \"<your query text>\"")
            sys.exit(1)
        query = sys.argv[2]
        search(query)
    else:
        print(f"Unknown command: {command}")