# inspector.py

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration (should match main.py) ---
INDEX_FILE = "faiss_index.bin"
DATA_FILE = "text_data.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'

def inspect_vector(index_id):
    """
    Retrieves and displays the text for a specific vector index.
    """
    with open(DATA_FILE, 'rb') as f:
        chunk_metadata = pickle.load(f)

    if 0 <= index_id < len(chunk_metadata):
        metadata = chunk_metadata[index_id]
        print(f"--- Data for Vector Index: {index_id} ---")
        print(f"Source PDF: {metadata['source']}")
        print("--- Text Chunk ---")
        print(metadata['text'])
        print("--------------------")
    else:
        print(f"Error: Index {index_id} is out of bounds.")


def find_and_inspect(query_text, k=5):
    """
    Performs a search and shows the results with their vector index IDs.
    """
    print(f"Running search for: '{query_text}'")
    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    
    query_vector = model.encode([query_text])
    distances, indices = index.search(np.array(query_vector), k)

    print("\n--- Search Inspection ---")
    for i, idx in enumerate(indices[0]):
        print(f"\nResult #{i+1}: Matched Vector Index ID: {idx}")
        inspect_vector(idx)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inspector.py <command> [value]")
        print("Commands:")
        print("  search <query> - Searches and shows matched vector IDs.")
        print("  get <index_id> - Gets the text for a specific vector ID.")
        sys.exit(1)

    command = sys.argv[1].lower()
    value = sys.argv[2]

    if command == 'search':
        find_and_inspect(value)
    elif command == 'get':
        try:
            index_id = int(value)
            inspect_vector(index_id)
        except ValueError:
            print("Error: Please provide a valid integer for the index ID.")
    else:
        print(f"Unknown command: {command}")