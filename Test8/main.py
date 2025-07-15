import os
import getpass
import config
from database import VectorStore
import json
import re
from typing import List, Dict, Any

def get_openai_key():
    api_key = config.OPENAI_API_KEY
    if not api_key:
        try:
            api_key = getpass.getpass("Please enter your OpenAI API key: ")
        except Exception as e:
            print(f"Could not read API key: {e}")
            return None
    return api_key

def process_pdf_workflow(filepath: str, db: VectorStore, image_service, embedding_service):
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

    processor = ContentProcessor(image_service=image_service)
    page_data_list, images_to_store = processor.process_content(raw_content, pdf_filename)

    if not page_data_list:
        print("No content could be processed from the PDF.")
        return

    page_summaries = [page_data['page_summary'] for page_data in page_data_list]
    embeddings = embedding_service.create_embeddings(page_summaries)

    for i, page_data in enumerate(page_data_list):
        page_data['embedding'] = embeddings[i]
        page_id = db.insert_page_data(page_data)
        
        for img in images_to_store:
            if img['page_number'] == page_data['page_number']:
                img['page_id'] = page_id

    if images_to_store:
        db.insert_images(images_to_store)

    print(f"\n✅ Successfully processed and stored '{pdf_filename}'.")

class TableQueryProcessor:
    def __init__(self):
        self.table_number_pattern = r'table\s+(\d+)'
        self.amount_patterns = [
            r'amount', r'value', r'income', r'cost', r'expense', r'investment', 
            r'revenue', r'profit', r'loss', r'balance', r'total'
        ]
        self.year_pattern = r'(19|20)\d{2}'
        
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        intent = {
            'table_number': None,
            'target_year': None,
            'target_metric': None,
            'table_keywords': [],
            'specific_value_request': False
        }
        
        # Extract table number
        table_match = re.search(self.table_number_pattern, query_lower)
        if table_match:
            intent['table_number'] = int(table_match.group(1))
        
        # Extract year
        year_match = re.search(self.year_pattern, query)
        if year_match:
            intent['target_year'] = year_match.group(0)
        
        # Extract metric type
        for pattern in self.amount_patterns:
            if re.search(pattern, query_lower):
                intent['target_metric'] = pattern
                intent['specific_value_request'] = True
                break
        
        # Extract table-related keywords
        table_keywords = ['financial', 'income', 'statement', 'year-end', 'accounts']
        for keyword in table_keywords:
            if keyword in query_lower:
                intent['table_keywords'].append(keyword)
        
        return intent

    def filter_relevant_tables(self, search_results: List[Dict], intent: Dict[str, Any]) -> List[Dict]:
        relevant_tables = []
        
        for result in search_results:
            try:
                if isinstance(result['json_metadata'], str):
                    metadata = json.loads(result['json_metadata'])
                else:
                    metadata = result['json_metadata']
                
                tables = [elem for elem in metadata['elements'] if elem['type'] == 'table']
                
                for table in tables:
                    table_name = table.get('table_name', '').lower()
                    
                    # Check for specific table number
                    if intent['table_number']:
                        if f"table {intent['table_number']}" in table_name:
                            relevant_tables.append({
                                'table': table,
                                'source': result['source_pdf'],
                                'page': result['page_number'],
                                'priority': 10  # Highest priority for exact table match
                            })
                            continue
                    
                    # Check for table keywords
                    priority = 0
                    for keyword in intent['table_keywords']:
                        if keyword in table_name:
                            priority += 2
                    
                    if priority > 0 or not intent['table_number']:
                        relevant_tables.append({
                            'table': table,
                            'source': result['source_pdf'],
                            'page': result['page_number'],
                            'priority': priority
                        })
                        
            except Exception as e:
                continue
        
        # Sort by priority and return top tables
        relevant_tables.sort(key=lambda x: x['priority'], reverse=True)
        return relevant_tables[:3]  # Return top 3 most relevant tables

    def extract_specific_value(self, table: Dict, intent: Dict[str, Any]) -> str:
        if not table.get('structure') or not table['structure'].get('rows'):
            return None
        
        rows = table['structure']['rows']
        headers = table['structure'].get('headers', [])
        
        # Look for the specific metric and year
        target_year = intent.get('target_year')
        target_metric = intent.get('target_metric')
        
        for row in rows:
            row_text = ' '.join([str(v) for v in row.values()]).lower()
            
            # Check if this row contains the target metric
            if target_metric and target_metric in row_text:
                # Look for the value in the target year column
                if target_year:
                    for header in headers:
                        if target_year in str(header):
                            value = row.get(header)
                            if value and str(value).replace(',', '').replace('(', '').replace(')', '').isdigit():
                                return str(value)
                
                # If no year specified, return the most likely numeric value
                for key, value in row.items():
                    if value and str(value).replace(',', '').replace('(', '').replace(')', '').replace('.', '').isdigit():
                        if len(str(value).replace(',', '')) > 2:  # Avoid small numbers like years
                            return str(value)
        
        return None

    def create_focused_context(self, relevant_tables: List[Dict], intent: Dict[str, Any]) -> str:
        context_parts = []
        
        for item in relevant_tables:
            table = item['table']
            table_name = table.get('table_name', 'Unknown Table')
            
            context_parts.append(f"TABLE: {table_name}")
            context_parts.append(f"Source: {item['source']}, Page {item['page']}")
            
            if table.get('structure') and table['structure'].get('rows'):
                headers = table['structure'].get('headers', [])
                context_parts.append(f"Columns: {', '.join(headers)}")
                
                # Show only relevant rows (first 5 to keep context short)
                relevant_rows = table['structure']['rows'][:5]
                for i, row in enumerate(relevant_rows):
                    row_desc = []
                    for header in headers:
                        value = row.get(header, '')
                        if value and str(value).strip():
                            row_desc.append(f"{header}: {value}")
                    if row_desc:
                        context_parts.append(f"  {' | '.join(row_desc)}")
            
            context_parts.append("")  # Empty line between tables
        
        return "\n".join(context_parts)

def query_workflow(query: str, db: VectorStore, embedding_service, llm_service):
    print(f"\n--- Searching for: '{query}' ---")

    # Initialize the table query processor
    processor = TableQueryProcessor()
    intent = processor.extract_query_intent(query)
    
    query_vector = embedding_service.create_embeddings([query])[0]
    search_results = db.search(query_vector, k=5)

    if not search_results:
        print("No relevant information found in the database.")
        return

    # Filter and get most relevant tables
    relevant_tables = processor.filter_relevant_tables(search_results, intent)
    
    if not relevant_tables:
        print("No relevant tables found for this query.")
        return
    
    # Try to extract specific value first
    if intent['specific_value_request'] and intent['table_number']:
        for item in relevant_tables:
            if item['priority'] >= 10:  # Exact table match
                specific_value = processor.extract_specific_value(item['table'], intent)
                if specific_value:
                    print(f"\n--- Final Answer ---")
                    print(f"{specific_value}")
                    print(f"\n--- Source ---")
                    print(f"- {item['table'].get('table_name', 'Unknown Table')}")
                    print(f"- {item['source']}, Page {item['page']}")
                    return

    # Create focused context for LLM
    context = processor.create_focused_context(relevant_tables, intent)
    
    # Create a precise prompt
    if intent['specific_value_request']:
        prompt = f"""You are a data extraction assistant. Extract the exact value requested from the table data below.

Question: {query}

Table Data:
{context}

INSTRUCTIONS:
1. Find the specific table mentioned in the question
2. Locate the exact row and column for the requested data
3. Return ONLY the numerical value (e.g., "200,000" or "654")
4. Do not include any explanation or additional text
5. If the value is not found, respond with "Value not found"

Answer:"""
    else:
        prompt = f"""Based on the table data below, answer this question concisely:

Question: {query}

Table Data:
{context}

Provide a clear, direct answer:"""

    final_answer = llm_service.generate_answer(query, prompt)

    print("\n--- Final Answer ---")
    print(final_answer.strip())
    print("\n--- Sources ---")
    for item in relevant_tables:
        print(f"- {item['table'].get('table_name', 'Unknown Table')}")
        print(f"  {item['source']}, Page {item['page']}")

def main():
    print("--- RAG PDF Processing and Query System ---")
    
    db = None
    try:
        print("\n--- Initializing and setting up database... ---")
        db = VectorStore(config.DB_CONFIG)
        db.connect()
        db.setup_database()
        print("--- Database setup successful. ---")

        api_key = get_openai_key()
        if not api_key:
            return

        print("\n--- Initializing AI services (this may take a moment)... ---")
        from services import ImageToTextService, EmbeddingService, LLMService
        image_service = ImageToTextService(api_key)
        embedding_service = EmbeddingService(config.EMBEDDING_MODEL)
        llm_service = LLMService(config.GENERATOR_MODEL)
        print("--- AI services initialized successfully. ---")

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
                break
            else:
                print("Invalid option. Please try again.")

    except Exception as e:
        print(f"An application-level error occurred: {e}")
    finally:
        if db:
            db.close()
        print("Goodbye!")

if __name__ == "__main__":
    if config.JAVA_HOME_PATH and not os.environ.get('JAVA_HOME'):
        os.environ['JAVA_HOME'] = config.JAVA_HOME_PATH
        
    main()