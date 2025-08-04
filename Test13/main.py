import os
import getpass
import config
from database import VectorStore
from extractor import UniversalExtractor, BatchExtractor
from processor import AdvancedTableAnalyzer, CSVProcessor
from services import ImageToTextService, EmbeddingService, LLMService, CSVQueryProcessor
import json
import re
import psycopg2.extras
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

def get_openai_key():
    api_key = config.OPENAI_API_KEY
    if not api_key:
        try:
            api_key = getpass.getpass("Please enter your OpenAI API key: ")
        except Exception as e:
            print(f"Could not read API key: {e}")
            return None
    return api_key

def get_azure_endpoint():
    """Get Azure endpoint from config or user input"""
    if hasattr(config, 'AZURE_ENDPOINT') and config.AZURE_ENDPOINT:
        return config.AZURE_ENDPOINT
    else:
        return input("Enter Azure OpenAI endpoint (or press Enter for standard OpenAI): ").strip() or None

def process_pdf_workflow(filepath: str, db: VectorStore, image_service, embedding_service, csv_processor):
    """Enhanced PDF processing workflow with CSV extraction"""
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    pdf_filename = os.path.basename(filepath)
    print(f"\n--- Starting to Process: {pdf_filename} ---")

    # Check if PDF already exists in database
    existing_count = check_existing_pdf(db, pdf_filename)
    if existing_count > 0:
        print(f"⚠️  PDF '{pdf_filename}' already exists in database ({existing_count} pages found)")
        while True:
            choice = input("Do you want to (r)eplace existing data, (c)ancel processing, or (s)kip and continue? [r/c/s]: ").strip().lower()
            if choice in ['r', 'replace']:
                print("🔄 Replacing existing data...")
                db.clear_documents(pdf_filename)
                break
            elif choice in ['c', 'cancel']:
                print("❌ Processing cancelled.")
                return
            elif choice in ['s', 'skip']:
                print("⏭️  Skipping processing (keeping existing data).")
                return
            else:
                print("Please enter 'r' for replace, 'c' for cancel, or 's' for skip.")
    else:
        db.clear_documents(pdf_filename)

    # Use universal extractor
    extractor = UniversalExtractor()
    extraction_result = extractor.extract_content(filepath)
    
    if not extraction_result["success"]:
        print(f"❌ Failed to extract content: {extraction_result['error']}")
        return
    
    raw_content = extraction_result["raw_content"]

    # Set up enhanced processor
    processor = AdvancedTableAnalyzer(image_service=image_service)
    processor.set_csv_processor(csv_processor)
    processor.db = db  # Set database reference for extraction logging
    
    page_data_list, images_to_store = processor.process_content(raw_content, pdf_filename)

    if not page_data_list:
        print("No content could be processed from the PDF.")
        return

    # Create embeddings for pages
    page_summaries = [page_data['page_summary'] for page_data in page_data_list]
    embeddings = embedding_service.create_embeddings(page_summaries)

    # Store page data
    for i, page_data in enumerate(page_data_list):
        page_data['embedding'] = embeddings[i]
        page_id = db.insert_page_data(page_data)
        
        # Update images with page_id
        for img in images_to_store:
            if img['page_number'] == page_data['page_number']:
                img['page_id'] = page_id

    # Store images
    if images_to_store:
        db.insert_images(images_to_store)

    # Show extraction summary
    extractions = db.get_table_extractions(pdf_filename)
    if extractions:
        print(f"\n📊 Table Extractions Summary:")
        for extraction in extractions:
            print(f"  - {extraction['table_name']} (Page {extraction['page_number']}) - {extraction['extraction_method']}")
    
    print(f"\n✅ Successfully processed and stored '{pdf_filename}'.")

def process_csv_workflow(filepath: str, db: VectorStore, embedding_service):
    """Process CSV file workflow"""
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    filename = os.path.basename(filepath)
    print(f"\n--- Starting to Process CSV: {filename} ---")

    # Check if CSV already exists
    existing_doc = db.get_csv_document_info(filename)
    if existing_doc:
        print(f"⚠️  CSV '{filename}' already exists in database")
        while True:
            choice = input("Do you want to (r)eplace existing data, (c)ancel processing, or (s)kip and continue? [r/c/s]: ").strip().lower()
            if choice in ['r', 'replace']:
                print("🔄 Replacing existing data...")
                db.clear_csv_document(filename)
                break
            elif choice in ['c', 'cancel']:
                print("❌ Processing cancelled.")
                return
            elif choice in ['s', 'skip']:
                print("⏭️  Skipping processing (keeping existing data).")
                return
            else:
                print("Please enter 'r' for replace, 'c' for cancel, or 's' for skip.")

    # Process CSV
    csv_processor = CSVProcessor(db, embedding_service)
    result = csv_processor.process_csv_file(filepath)
    
    if result['success']:
        print(f"✅ {result['message']}")
        
        # Show CSV info
        doc_info = db.get_csv_document_info(filename)
        if doc_info:
            print(f"📊 CSV Info:")
            print(f"  - Rows: {doc_info['total_rows']}")
            print(f"  - Columns: {len(doc_info['column_names'])}")
            print(f"  - Column names: {', '.join(doc_info['column_names'])}")
    else:
        print(f"❌ {result['message']}")

def batch_process_workflow(directory_path: str, db: VectorStore, image_service, embedding_service, csv_processor):
    """Process multiple files from directory"""
    
    print(f"\n--- Starting Batch Processing: {directory_path} ---")
    
    batch_extractor = BatchExtractor()
    extraction_results = batch_extractor.extract_from_directory(directory_path)
    
    for result in extraction_results:
        if not result['success']:
            print(f"❌ Failed to process {result.get('filepath', 'unknown file')}: {result['error']}")
            continue
        
        file_type = result['file_type']
        filepath = result['filepath']
        
        if file_type == 'pdf':
            # Process as PDF
            print(f"\n📄 Processing PDF: {result['filename']}")
            process_pdf_workflow(filepath, db, image_service, embedding_service, csv_processor)
        
        elif file_type == 'csv':
            # Process as CSV
            print(f"\n📊 Processing CSV: {result['filename']}")
            process_csv_workflow(filepath, db, embedding_service)

def check_existing_pdf(db: VectorStore, pdf_filename: str) -> int:
    """Check if PDF already exists in database and return count of pages"""
    try:
        db._ensure_connection()
        with db.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_pages WHERE source_pdf = %s;", (pdf_filename,))
            count = cur.fetchone()[0]
        return count
    except Exception as e:
        print(f"Error checking existing PDF: {e}")
        return 0

# Enhanced query processor (existing logic with CSV support)
class StructureAwareQueryProcessor:
    def __init__(self):
        self.table_number_pattern = r'table\s+(\d+)'
        self.person_pattern = r'\b([A-Z][a-z]+)\b'
        self.year_pattern = r'\b(19|20)\d{2}\b'
        
        self.continent_patterns = [
            'africa', 'asia', 'europe', 'america', 'australia', 'south america'
        ]
        
        self.metric_patterns = [
            'highest', 'average', 'hour', 'high', 'peak', 'maximum', 'minimum'
        ]
        
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        intent = {
            'table_number': None,
            'person_name': None,
            'continent': None,
            'metric': None,
            'year': None,
            'action': None,
            'query_type': 'unknown',
            'specific_table_request': False,
            'csv_request': 'csv' in query_lower or 'convert' in query_lower
        }
        
        # Extract table number
        table_match = re.search(self.table_number_pattern, query_lower)
        if table_match:
            intent['table_number'] = int(table_match.group(1))
            intent['specific_table_request'] = True
        
        # Extract year
        year_match = re.search(self.year_pattern, query)
        if year_match:
            intent['year'] = year_match.group(0)
        
        # Check for continent-based query
        for continent in self.continent_patterns:
            if continent in query_lower:
                intent['continent'] = continent.replace(' ', '_')
                intent['query_type'] = 'continent_metric'
                break
        
        # Check for metric patterns
        metric_phrase = None
        if "24 hour" in query_lower:
            metric_phrase = "24 hour"
            intent['metric'] = "highest in 24 hours"
        elif "12 hour" in query_lower:
            metric_phrase = "12 hour"
            intent['metric'] = "highest in 12 hours"
        elif "average" in query_lower:
            intent['metric'] = "average"
        else:
            for metric in self.metric_patterns:
                if metric in query_lower:
                    intent['metric'] = metric
                    if intent['query_type'] == 'unknown':
                        intent['query_type'] = 'metric_based'
                    break
        
        # Check for person-based query
        if intent['query_type'] == 'unknown':
            person_matches = re.findall(self.person_pattern, query)
            for person in person_matches:
                if person.lower() not in ['africa', 'asia', 'europe', 'america', 'australia']:
                    intent['person_name'] = person.lower()
                    intent['query_type'] = 'person_action'
                    break
        
        # Extract action for person-based queries
        if intent['query_type'] == 'person_action':
            if 'won' in query_lower:
                intent['action'] = 'won'
            elif 'entered' in query_lower:
                intent['action'] = 'entered'
            elif 'completed' in query_lower:
                intent['action'] = 'completed'
        
        return intent

    def find_exact_table(self, search_results: List[Dict], intent: Dict[str, Any]) -> Optional[Dict]:
        """Find the exact table requested"""
        target_table_num = intent.get('table_number')
        
        if not target_table_num:
            return None
        
        print(f"🎯 Looking for Table {target_table_num}")
        
        for result in search_results:
            try:
                if isinstance(result['json_metadata'], str):
                    metadata = json.loads(result['json_metadata'])
                else:
                    metadata = result['json_metadata']
                
                tables = [elem for elem in metadata['elements'] if elem['type'] == 'table']
                
                for table in tables:
                    table_name = table.get('table_name', '').lower()
                    if f"table {target_table_num}" in table_name:
                        print(f"✅ Found {table.get('table_name')}")
                        return {
                            'table': table,
                            'source': result['source_pdf'],
                            'page': result['page_number']
                        }
                        
            except Exception as e:
                continue
        
        return None

    def extract_value_by_structure(self, table_data: Dict, intent: Dict[str, Any]) -> Optional[str]:
        """Extract value based on dynamic table structure"""
        table = table_data['table']
        structure = table.get('structure', {})
        structure_type = structure.get('structure_type', 'unknown')
        
        print(f"\n🎯 DYNAMIC STRUCTURE EXTRACTION")
        print(f"📊 Table: {table.get('table_name')}")
        print(f"🏗️ Structure type: {structure_type}")
        print(f"❓ Query type: {intent.get('query_type')}")
        
        rows = structure.get('rows', [])
        
        # Dynamic extraction based on intent and structure
        return self._extract_best_match(rows, intent)

    def _extract_best_match(self, rows: List[Dict], intent: Dict[str, Any]) -> Optional[str]:
        """Find best matching value using dynamic approach"""
        
        # Collect search terms from intent
        search_terms = []
        if intent.get('person_name'):
            search_terms.append(intent['person_name'].lower())
        if intent.get('continent'):
            search_terms.append(intent['continent'].replace('_', ' ').lower())
        if intent.get('year'):
            search_terms.append(intent['year'])
        if intent.get('metric'):
            search_terms.append(intent['metric'].lower())
        if intent.get('action'):
            search_terms.append(intent['action'].lower())
        
        print(f"🔍 Search terms: {search_terms}")
        
        best_match = None
        best_score = 0
        
        for row in rows:
            # Calculate match score for this row
            score = 0
            matched_terms = []
            
            # Check all values in the row for matches
            row_text = ' '.join(str(v) for v in row.values()).lower()
            
            for term in search_terms:
                if term in row_text:
                    score += 1
                    matched_terms.append(term)
            
            # Boost score for exact key matches
            for key, value in row.items():
                key_lower = key.lower()
                value_lower = str(value).lower()
                
                for term in search_terms:
                    if term in key_lower or term in value_lower:
                        score += 0.5
            
            print(f"Row match: {score:.1f} - {matched_terms} - {list(row.values())[:3]}")
            
            if score > best_score:
                best_score = score
                best_match = row
        
        if best_match and best_score >= 1:  # At least one term must match
            # Find the most relevant value to return
            return self._extract_relevant_value(best_match, intent)
        
        return None

    def _extract_relevant_value(self, row: Dict, intent: Dict[str, Any]) -> Optional[str]:
        """Extract the most relevant value from the matched row"""
        
        # Look for numeric values first (most common query result)
        numeric_values = []
        for key, value in row.items():
            str_value = str(value).strip()
            # Check if it's a number (with or without decimals)
            if re.match(r'^\d+\.?\d*$', str_value):
                numeric_values.append((key, str_value))
        
        if numeric_values:
            # If we have specific intent, try to match the key
            if intent.get('action'):
                action = intent['action'].lower()
                for key, value in numeric_values:
                    if action in key.lower():
                        return value
            
            # Return the first numeric value found
            return numeric_values[0][1]
        
        # If no numeric values, return the most relevant text value
        for key, value in row.items():
            if key.lower() not in ['identifier', 'name', 'metric', 'year', 'category']:
                str_value = str(value).strip()
                if str_value and str_value.lower() != 'nan':
                    return str_value
        
        return None

    # Extended Context Methods
    def get_full_page_content(self, db, source_pdf, page_number):
        """Get complete page content from database"""
        try:
            db._ensure_connection()
            with db.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT id, html_content, json_metadata, page_summary
                    FROM document_pages 
                    WHERE source_pdf = %s AND page_number = %s;
                """, (source_pdf, page_number))
                
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            print(f"❌ Error getting full page content: {e}")
            return None

    def get_related_pages(self, db, source_pdf, current_page, k=5):
        """Get related pages from the same document"""
        try:
            db._ensure_connection()
            with db.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute("""
                    SELECT page_number, page_summary, json_metadata, html_content
                    FROM document_pages 
                    WHERE source_pdf = %s 
                    AND page_number != %s
                    ORDER BY ABS(page_number - %s)  -- Get pages closest to current page
                    LIMIT %s;
                """, (source_pdf, current_page, current_page, k))
                
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            print(f"❌ Error getting related pages: {e}")
            return []

    def extract_text_from_html(self, html_content):
        """Extract clean text from HTML content"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            print(f"⚠️  BeautifulSoup error: {e}, using simple method")
            # Fallback: simple HTML tag removal
            import re
            text = re.sub('<[^<]+?>', '', html_content)
            return text.strip()

    def create_expanded_context(self, result, query, db, embedding_service, k_additional=3):
        """Create expanded context by pulling related pages and full content"""
        
        context_parts = []
        
        if result['source_type'] == 'pdf':
            try:
                # 1. Get the current page details
                source_pdf = result.get('filename') or result.get('source_pdf')
                page_number = result.get('page_number')
                
                print(f"🔍 Expanding context from {source_pdf}, Page {page_number}")
                
                # 2. Get full page data from database
                full_page_data = self.get_full_page_content(db, source_pdf, page_number)
                
                if full_page_data:
                    # Add full HTML content (not truncated)
                    if full_page_data.get('html_content'):
                        clean_text = self.extract_text_from_html(full_page_data['html_content'])
                        context_parts.append(f"PAGE {page_number} CONTENT:\n{clean_text}")
                        print(f"✅ Added page content: {len(clean_text)} characters")
                    
                    # Add full JSON metadata with all table data
                    if full_page_data.get('json_metadata'):
                        try:
                            metadata = json.loads(full_page_data['json_metadata']) if isinstance(full_page_data['json_metadata'], str) else full_page_data['json_metadata']
                            
                            # Extract all table data comprehensively
                            tables = [elem for elem in metadata.get('elements', []) if elem['type'] == 'table']
                            
                            if tables:
                                context_parts.append("\nTABLE DATA FROM CURRENT PAGE:")
                                for table_idx, table in enumerate(tables):
                                    table_name = table.get('table_name', f'Table {table_idx + 1}')
                                    context_parts.append(f"\n{table_name}:")
                                    
                                    # Get ALL rows from the table
                                    if 'structure' in table and 'rows' in table['structure']:
                                        rows = table['structure']['rows']
                                        print(f"✅ Found table with {len(rows)} rows")
                                        
                                        for row_idx, row in enumerate(rows):
                                            row_desc = []
                                            for key, value in row.items():
                                                if value and str(value).strip() and str(value).lower() != 'nan':
                                                    row_desc.append(f"{key}: {value}")
                                            if row_desc:
                                                context_parts.append(f"  Row {row_idx + 1}: {' | '.join(row_desc)}")
                                    
                                    # Add searchable text
                                    if 'searchable_text' in table:
                                        context_parts.append(f"  Summary: {table['searchable_text']}")
                        except Exception as e:
                            print(f"⚠️  Error parsing JSON metadata: {e}")
                
                # 3. Get related pages from the same document
                print(f"🔍 Getting {k_additional} related pages...")
                related_pages = self.get_related_pages(db, source_pdf, page_number, k_additional)
                
                for related_page in related_pages:
                    context_parts.append(f"\nRELATED PAGE {related_page['page_number']}:")
                    
                    # Add page summary
                    if related_page.get('page_summary'):
                        context_parts.append(f"Summary: {related_page['page_summary']}")
                    
                    # Add HTML content from related pages
                    if related_page.get('html_content'):
                        related_text = self.extract_text_from_html(related_page['html_content'])
                        if len(related_text) > 100:  # Only add if substantial content
                            context_parts.append(f"Content: {related_text[:1000]}...")  # Limit each related page
                    
                    # Add table data from related pages
                    if related_page.get('json_metadata'):
                        try:
                            rel_metadata = json.loads(related_page['json_metadata'])
                            rel_tables = [elem for elem in rel_metadata.get('elements', []) if elem['type'] == 'table']
                            
                            for table in rel_tables:
                                if 'structure' in table and 'rows' in table['structure']:
                                    table_name = table.get('table_name', 'Related Table')
                                    context_parts.append(f"  {table_name}:")
                                    
                                    # Add financial data from related tables (limit to 3 rows per related table)
                                    for row in table['structure']['rows'][:3]:
                                        row_text = ' | '.join(f"{k}: {v}" for k, v in row.items() 
                                                            if v and str(v).strip() and str(v).lower() != 'nan')
                                        if row_text:
                                            context_parts.append(f"    {row_text}")
                        except Exception as e:
                            print(f"⚠️  Error parsing related page metadata: {e}")
            
            except Exception as e:
                print(f"⚠️  Error creating expanded context: {e}")
                # Fallback to basic content
                context_parts.append(f"BASIC CONTENT: {result.get('content', '')}")
        
        elif result['source_type'] == 'csv':
            context_parts.append(f"CSV Data from {result['filename']}: {result['content']}")
        
        # Join all context parts
        full_context = "\n".join(context_parts)
        
        print(f"✅ Expanded context created: {len(full_context)} characters")
        return full_context

    def query_workflow_enhanced(self, query: str, db: VectorStore, embedding_service, llm_service, csv_query_processor):
        """Enhanced query workflow with confidence scoring and expanded context"""
        print(f"\n--- Enhanced Search for: '{query}' ---")

        processor = StructureAwareQueryProcessor()
        intent = processor.extract_query_intent(query)
        
        print(f"🎯 Query intent: {intent}")
        
        # Handle CSV-specific queries
        if intent.get('csv_request'):
            print("🔍 Searching CSV data...")
            csv_results = csv_query_processor.semantic_search(query, k=5)
            
            if csv_results:
                print(f"\n--- CSV Results (Top Result Only) ---")
                # Return only the highest confidence result
                best_result = max(csv_results, key=lambda x: x['similarity'])
                print(f"File: {best_result['filename']}")
                print(f"Row: {best_result['row_index']}")
                print(f"Confidence: {best_result['similarity']:.3f}")
                print(f"Data: {best_result['row_data']}")
                return
        
        # Enhanced PDF + CSV search with confidence scoring
        query_vector = embedding_service.create_single_embedding(query)
        search_results = db.search_combined(query_vector, k=10)
        
        if not search_results:
            print("No relevant information found in the database.")
            return

        # Score and rank all results
        scored_results = []
        
        for result in search_results:
            confidence_score = result['similarity']
            
            # Boost score for specific matches
            if intent['specific_table_request']:
                if result['source_type'] == 'pdf':
                    try:
                        metadata = json.loads(result['json_metadata']) if isinstance(result['json_metadata'], str) else result['json_metadata']
                        tables = [elem for elem in metadata['elements'] if elem['type'] == 'table']
                        
                        for table in tables:
                            table_name = table.get('table_name', '').lower()
                            if f"table {intent['table_number']}" in table_name:
                                confidence_score += 0.3  # Boost for exact table match
                                
                                # Try structure-aware extraction
                                table_data = {
                                    'table': table,
                                    'source': result['filename'],
                                    'page': result['page_number']
                                }
                                direct_value = processor.extract_value_by_structure(table_data, intent)
                                if direct_value:
                                    scored_results.append({
                                        'result': result,
                                        'confidence': confidence_score + 0.2,  # Extra boost for successful extraction
                                        'answer': direct_value,
                                        'source': f"{table.get('table_name')} - {result['filename']}, Page {result['page_number']}",
                                        'type': 'direct_extraction'
                                    })
                                    continue
                    except:
                        pass
            
            # Add to scored results for LLM processing
            scored_results.append({
                'result': result,
                'confidence': confidence_score,
                'answer': None,
                'source': f"{result['filename']} - {result.get('page_number', 'N/A')}",
                'type': 'similarity_match'
            })
        
        # Sort by confidence and take the best result
        scored_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        if not scored_results:
            print("No relevant results found.")
            return
        
        best_result = scored_results[0]
        
        print(f"\n--- Highest Confidence Result ---")
        print(f"Confidence Score: {best_result['confidence']:.3f}")
        print(f"Source: {best_result['source']}")
        
        if best_result['answer']:
            # Direct extraction was successful
            print(f"Answer: {best_result['answer']}")
            print(f"Method: Direct Structure Extraction")
        else:
            # Use LLM with expanded context
            result = best_result['result']
            
            print(f"\n🔍 Creating expanded context...")
            
            # Use the expanded context method
            context = self.create_expanded_context(result, query, db, embedding_service, k_additional=3)
            
            # If context is still too short, try to get more pages
            if len(context) < 1000:
                print("⚠️  Context still short, trying to get more pages...")
                context = self.create_expanded_context(result, query, db, embedding_service, k_additional=8)
            
            # DEBUG: Show what context is being sent to LLM
            print(f"\n🔍 DEBUG - Expanded Context:")
            print(f"Context length: {len(context)} characters")
            print(f"Context preview (first 500 chars): {context[:500]}...")
            if len(context) > 1000:
                print(f"Context preview (last 500 chars): ...{context[-500:]}")
            
            # Show sample of middle content to verify tables are included
            if len(context) > 2000:
                mid_point = len(context) // 2
                print(f"Context middle sample: ...{context[mid_point-250:mid_point+250]}...")
            
            # Generate answer with GPT-4o
            print(f"\n🤖 Sending expanded context to GPT-4o...")
            answer = llm_service.generate_answer(query, context)
            print(f"Answer: {answer}")
            print(f"Method: GPT-4o Analysis with Expanded Context")
        
        print(f"\nConfidence Level: {'High' if best_result['confidence'] > 0.8 else 'Medium' if best_result['confidence'] > 0.5 else 'Low'}")

# Update the main query workflow function
def query_workflow(query: str, db: VectorStore, embedding_service, llm_service, csv_query_processor):
    """Enhanced query workflow with confidence scoring and expanded context"""
    processor = StructureAwareQueryProcessor()
    enhanced_workflow = processor.query_workflow_enhanced(query, db, embedding_service, llm_service, csv_query_processor)

def show_database_stats(db: VectorStore):
    """Show database statistics"""
    print("\n--- Database Statistics ---")
    
    # PDF documents
    db._ensure_connection()
    with db.conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT source_pdf) FROM document_pages;")
        pdf_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM document_pages;")
        page_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM document_images;")
        image_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM csv_documents;")
        csv_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM csv_rows;")
        csv_row_count = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM table_extractions;")
        extraction_count = cur.fetchone()[0]
    
    print(f"📄 PDF Documents: {pdf_count}")
    print(f"📝 Pages: {page_count}")
    print(f"🖼️ Images: {image_count}")
    print(f"📊 CSV Documents: {csv_count}")
    print(f"📋 CSV Rows: {csv_row_count}")
    print(f"🔍 Table Extractions: {extraction_count}")

def main():
    print("--- Enhanced RAG PDF & CSV Processing System ---")
    
    db = None
    try:
        print("\n--- Initializing database... ---")
        db = VectorStore(config.DB_CONFIG)
        db.connect()
        db.setup_database()
        db.optimize_indexes()
        print("--- Database setup successful. ---")

        api_key = get_openai_key()
        if not api_key:
            return
        
        azure_endpoint = get_azure_endpoint()

        print("\n--- Initializing AI services... ---")
        image_service = ImageToTextService(api_key, azure_endpoint)
        embedding_service = EmbeddingService(config.EMBEDDING_MODEL)
        llm_service = LLMService(api_key, azure_endpoint, config.GENERATOR_MODEL)
        csv_processor = CSVProcessor(db, embedding_service)
        csv_query_processor = CSVQueryProcessor(db, embedding_service)
        print("--- AI services initialized successfully. ---")

        while True:
            print("\n--- Main Menu ---")
            print("1. Process a PDF file")
            print("2. Process a CSV file")
            print("3. Batch process directory")
            print("4. Ask a question")
            print("5. Show database statistics")
            print("6. List CSV documents")
            print("7. Exit")
            choice = input("Select an option: ").strip()

            if choice == '1':
                filepath = input("Enter the full path to your PDF file: ").strip()
                process_pdf_workflow(filepath, db, image_service, embedding_service, csv_processor)
            
            elif choice == '2':
                filepath = input("Enter the full path to your CSV file: ").strip()
                process_csv_workflow(filepath, db, embedding_service)
            
            elif choice == '3':
                directory = input("Enter the directory path: ").strip()
                batch_process_workflow(directory, db, image_service, embedding_service, csv_processor)
            
            elif choice == '4':
                query = input("Enter your question: ").strip()
                if query:
                    query_workflow(query, db, embedding_service, llm_service, csv_query_processor)
            
            elif choice == '5':
                show_database_stats(db)
            
            elif choice == '6':
                csv_docs = db.get_csv_documents_list()
                if csv_docs:
                    print("\n--- CSV Documents ---")
                    for doc in csv_docs:
                        print(f"📊 {doc['filename']}")
                        print(f"   Rows: {doc['total_rows']}")
                        print(f"   Columns: {len(doc['column_names'])}")
                        print(f"   Type: {doc['source_type']}")
                        print(f"   Date: {doc['upload_date']}")
                        print()
                else:
                    print("No CSV documents found.")
            
            elif choice == '7':
                break
            else:
                print("Invalid option. Please try again.")

    except Exception as e:
        print(f"An application-level error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if db:
            db.close()
        print("Goodbye!")

if __name__ == "__main__":
    if config.JAVA_HOME_PATH and not os.environ.get('JAVA_HOME'):
        os.environ['JAVA_HOME'] = config.JAVA_HOME_PATH
        
    main()