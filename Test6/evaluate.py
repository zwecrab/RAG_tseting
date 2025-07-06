import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

def evaluate_retrieval():
    """
    Evaluates the search performance using a tabular evaluation set
    and calculates the Mean Reciprocal Rank (MRR).
    """
    eval_file = 'test/evaluation_set.json'
    index_file = 'faiss_index.bin'
    data_file = 'text_data.pkl'
    model_name = 'all-MiniLM-L6-v2'
    top_k = 10 # How many results to check for the correct answer

    # --- 1. Load Evaluation Set ---
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file '{eval_file}' not found.")
        return
    with open(eval_file, 'r', encoding='utf-8') as f:
        evaluation_set = json.load(f)

    # --- 2. Load Model and Index ---
    if not os.path.exists(index_file) or not os.path.exists(data_file):
        print("Error: Index files not found. Please run 'python main.py index' first.")
        return
        
    print("Loading model and index for evaluation...")
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_file)
    with open(data_file, 'rb') as f:
        chunk_metadata = pickle.load(f)
    print("Model and index loaded.")

    reciprocal_ranks = []
    
    print(f"\n--- Running evaluation on {len(evaluation_set)} questions ---")

    for item in evaluation_set:
        question = item['question']
        expected_doc = item['expected_document']
        expected_snippet = item['expected_snippet']
        
        print(f"\nEvaluating Q{item['id']}: \"{question}\"")
        
        # --- 3. Perform Search ---
        query_vector = model.encode([question])
        _distances, indices = index.search(np.array(query_vector), k=top_k)
        
        results = [chunk_metadata[idx] for idx in indices[0]]
        
        # --- 4. Find Rank ---
        rank = 0
        found = False
        for k, res in enumerate(results):
            # Check if both the document and the snippet match
            correct_doc = res['source'] == expected_doc
            correct_snippet = expected_snippet.lower() in res['text'].lower()
            
            if correct_doc and correct_snippet:
                rank = k + 1
                found = True
                break
        
        # --- 5. Calculate Reciprocal Rank for this query ---
        if found:
            reciprocal_rank = 1 / rank
            print(f"✅ Found correct answer at rank: {rank}")
            reciprocal_ranks.append(reciprocal_rank)
        else:
            print(f"❌ Correct answer not found in top {top_k} results.")
            reciprocal_ranks.append(0)

    # --- 6. Calculate Final MRR Score ---
    if reciprocal_ranks:
        mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks)
        print(f"\n--- Evaluation Complete ---")
        print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    else:
        print("Evaluation could not be completed.")

if __name__ == '__main__':
    evaluate_retrieval()