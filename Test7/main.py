# main.py

import os
import getpass
import config
from database import VectorStore
# AI-related imports are now moved inside the main() function

def get_openai_key():
    """Gets the OpenAI API key from config or prompts the user."""
    api_key = config.OPENAI_API_KEY
    if not api_key:
        try:
            api_key = getpass.getpass("Please enter your OpenAI API key: ")
        except Exception as e:
            print(f"Could not read API key: {e}")
            return None
    return api_key

def process_pdf_workflow(filepath: str, db: VectorStore, image_service, embedding_service):
    """
    Handles the entire workflow for processing a single PDF.
    Note: AI service types are not hinted here as they are imported dynamically.
    """
    # Dynamically import processor and extractor here to ensure they are loaded when needed
    from extractor import PDFExtractor
    from processor import ContentProcessor

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    pdf_filename = os.path.basename(filepath)
    print(f"\n--- Starting to Process: {pdf_filename} ---")

    db.clear_documents(pdf_filename)

    extractor = PDFExtractor()
    raw_content = extractor.extract_content(filepath)

    processor = ContentProcessor(
        image_service=image_service,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    final_chunks = processor.process_content(raw_content, pdf_filename)

    if not final_chunks:
        print("No content could be processed from the PDF.")
        return

    texts_to_embed = [chunk['content_text'] for chunk in final_chunks]
    embeddings = embedding_service.create_embeddings(texts_to_embed)

    for i, chunk in enumerate(final_chunks):
        chunk['embedding'] = embeddings[i]

    db.insert_chunks(final_chunks)
    print(f"\n✅ Successfully processed and stored '{pdf_filename}'.")

def query_workflow(query: str, db: VectorStore, embedding_service, llm_service):
    """
    Handles the workflow for answering a user query.
    Note: AI service types are not hinted here as they are imported dynamically.
    """
    print(f"\n--- Searching for: '{query}' ---")

    query_vector = embedding_service.create_embeddings([query])[0]
    search_results = db.search(query_vector, k=5)

    if not search_results:
        print("No relevant information found in the database.")
        return

    context = "\n---\n".join([f"Source: {res['source_pdf']} (Page {res['page_number']})\nContent: {res['content_text']}" for res in search_results])
    
    final_answer = llm_service.generate_answer(query, context)

    print("\n--- Final Answer ---")
    print(final_answer)
    print("\n--- Sources ---")
    for res in search_results:
        print(f"- {res['source_pdf']}, Page {res['page_number']} (Similarity: {res['similarity']:.4f})")


def main():
    """Main application loop."""
    print("--- RAG PDF Processing and Query System ---")
    
    db = None # Initialize db to None
    try:
        # --- STEP 1: Database Setup FIRST ---
        # We do this before initializing any heavy AI libraries to avoid conflicts.
        print("\n--- Initializing and setting up database... ---")
        db = VectorStore(config.DB_CONFIG)
        db.connect()
        db.setup_database()
        print("--- Database setup successful. ---")

        # --- STEP 2: Get API Key ---
        api_key = get_openai_key()
        if not api_key:
            return

        # --- STEP 3: DYNAMICALLY Initialize AI Services ---
        # Now that the database is confirmed to be working, load the AI models.
        # By importing here, we prevent conflicts during initial startup.
        print("\n--- Initializing AI services (this may take a moment)... ---")
        from services import ImageToTextService, EmbeddingService, LLMService
        image_service = ImageToTextService(api_key)
        embedding_service = EmbeddingService(config.EMBEDDING_MODEL)
        llm_service = LLMService(config.GENERATOR_MODEL)
        print("--- AI services initialized successfully. ---")

        # --- STEP 4: Main UI Loop ---
        while True:
            print("\n--- Main Menu ---")
            print("1. Process a new PDF")
            print("2. Ask a question")
            print("3. Exit")
            choice = input("Select an option: ").strip()

            if choice == '1':
                filepath = input("Enter the full path to your PDF file: ").strip()
                process_pdf_workflow(filepath, db, image_service, embedding_service)
            elif choice == '2':
                query = input("Enter your question: ").strip()
                if query:
                    query_workflow(query, db, embedding_service, llm_service)
            elif choice == '3':
                break # Exit the loop to reach the finally block
            else:
                print("Invalid option. Please try again.")

    except Exception as e:
        print(f"An application-level error occurred: {e}")
    finally:
        # --- Cleanup ---
        if db:
            db.close() # Ensure the connection is closed on exit
        print("Goodbye!")


if __name__ == "__main__":
    if config.JAVA_HOME_PATH and not os.environ.get('JAVA_HOME'):
        os.environ['JAVA_HOME'] = config.JAVA_HOME_PATH
        
    main()
