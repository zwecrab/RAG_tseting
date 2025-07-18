import psycopg2
import psycopg2.extras
import numpy as np
import hashlib
from typing import List, Dict, Optional
import json

class VectorStore:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None

    def connect(self):
        if self.conn and not self.conn.closed:
            return

        try:
            print("--- Establishing persistent database connection... ---")
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ Persistent database connection established.")
        except psycopg2.OperationalError as e:
            print(f"❌ Could not connect to the database. Please check config.py and ensure the server is running.")
            print(f"   Error details: {e}")
            raise

    def close(self):
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("Database connection closed.")

    def _ensure_connection(self):
        if self.conn is None or self.conn.closed:
            print("Connection lost. Reconnecting...")
            self.connect()

    def setup_database(self):
        """Setup all database tables including CSV-specific ones"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Original tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_pages (
                    id SERIAL PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    page_number INT NOT NULL,
                    html_content TEXT,
                    json_metadata JSONB,
                    page_summary TEXT,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_images (
                    id SERIAL PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    page_number INT NOT NULL,
                    image_index INT NOT NULL,
                    image_data BYTEA,
                    image_format VARCHAR(10),
                    image_hash VARCHAR(64),
                    image_type VARCHAR(50),
                    width INTEGER,
                    height INTEGER,
                    position_x REAL,
                    position_y REAL,
                    ai_description TEXT,
                    page_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # New CSV-specific tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS csv_documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_path TEXT,
                    upload_date TIMESTAMP DEFAULT NOW(),
                    total_rows INTEGER,
                    column_names TEXT[],
                    column_types JSONB,
                    file_hash VARCHAR(64) UNIQUE,
                    metadata JSONB,
                    source_type VARCHAR(20) DEFAULT 'upload' -- 'upload', 'extracted', 'pdf_table'
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS csv_rows (
                    id SERIAL PRIMARY KEY,
                    csv_doc_id INTEGER REFERENCES csv_documents(id) ON DELETE CASCADE,
                    row_index INTEGER NOT NULL,
                    row_data JSONB,
                    row_text TEXT,
                    embedding VECTOR(384),
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS csv_columns (
                    id SERIAL PRIMARY KEY,
                    csv_doc_id INTEGER REFERENCES csv_documents(id) ON DELETE CASCADE,
                    column_name TEXT NOT NULL,
                    column_type TEXT,
                    sample_values TEXT[],
                    column_description TEXT,
                    embedding VECTOR(384)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS table_extractions (
                    id SERIAL PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    page_number INT NOT NULL,
                    table_name TEXT,
                    extraction_method VARCHAR(50), -- 'tabula', 'vision_api', 'hybrid'
                    csv_content TEXT,
                    csv_doc_id INTEGER REFERENCES csv_documents(id),
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes for performance
            cur.execute("CREATE INDEX IF NOT EXISTS doc_page_embedding_idx ON document_pages USING hnsw (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS doc_images_hash_idx ON document_images (image_hash);")
            cur.execute("CREATE INDEX IF NOT EXISTS csv_rows_embedding_idx ON csv_rows USING hnsw (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS csv_columns_embedding_idx ON csv_columns USING hnsw (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS csv_rows_doc_idx ON csv_rows(csv_doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS csv_rows_data_gin_idx ON csv_rows USING gin (row_data);")
            cur.execute("CREATE INDEX IF NOT EXISTS csv_docs_filename_idx ON csv_documents(filename);")
            cur.execute("CREATE INDEX IF NOT EXISTS table_extractions_pdf_idx ON table_extractions(source_pdf, page_number);")
            
            self.conn.commit()
        print("✅ Database setup complete with CSV support.")

    def clear_documents(self, source_pdf: str):
        """Clear existing documents and related CSV data"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            # Clear related table extractions
            cur.execute("DELETE FROM table_extractions WHERE source_pdf = %s;", (source_pdf,))
            
            # Clear original document data
            cur.execute("DELETE FROM document_images WHERE source_pdf = %s;", (source_pdf,))
            cur.execute("DELETE FROM document_pages WHERE source_pdf = %s;", (source_pdf,))
            
            self.conn.commit()
        print(f"✅ Cleared existing documents for '{source_pdf}'")

    def clear_csv_document(self, filename: str):
        """Clear specific CSV document"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM csv_documents WHERE filename = %s;", (filename,))
            self.conn.commit()
        print(f"✅ Cleared CSV document '{filename}'")

    # Original methods
    def insert_page_data(self, page_data: Dict):
        self._ensure_connection()
        embedding_str = str(page_data['embedding'].tolist())
        
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_pages (source_pdf, page_number, html_content, json_metadata, page_summary, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector) RETURNING id;
                """,
                (
                    page_data['source_pdf'],
                    page_data['page_number'],
                    page_data['html_content'],
                    psycopg2.extras.Json(page_data['json_metadata']),
                    page_data['page_summary'],
                    embedding_str
                )
            )
            page_id = cur.fetchone()[0]
            self.conn.commit()
        return page_id

    def insert_images(self, images: List[Dict]):
        if not images:
            return
            
        self._ensure_connection()
        
        data_to_insert = []
        for img in images:
            image_hash = hashlib.sha256(img['image_data']).hexdigest()
            data_to_insert.append({
                'source_pdf': img['source_pdf'],
                'page_number': img['page_number'],
                'image_index': img['image_index'],
                'image_data': img['image_data'],
                'image_format': img.get('image_format', 'png'),
                'image_hash': image_hash,
                'image_type': img['image_type'],
                'width': img.get('width'),
                'height': img.get('height'),
                'position_x': img.get('position_x'),
                'position_y': img.get('position_y'),
                'ai_description': img.get('ai_description'),
                'page_id': img.get('page_id')
            })

        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO document_images 
                (source_pdf, page_number, image_index, image_data, image_format, image_hash, 
                 image_type, width, height, position_x, position_y, ai_description, page_id)
                VALUES (%(source_pdf)s, %(page_number)s, %(image_index)s, %(image_data)s, 
                        %(image_format)s, %(image_hash)s, %(image_type)s, %(width)s, %(height)s,
                        %(position_x)s, %(position_y)s, %(ai_description)s, %(page_id)s);
                """,
                data_to_insert
            )
            self.conn.commit()
        print(f"Successfully inserted {len(images)} images into the database.")

    # New CSV-specific methods
    def insert_csv_document(self, filename: str, file_path: str, headers: List[str], 
                           total_rows: int, file_hash: str, column_types: Dict = None,
                           source_type: str = 'upload', metadata: Dict = None) -> int:
        """Insert CSV document metadata with proper error handling"""
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO csv_documents (filename, file_path, total_rows, column_names, 
                                             column_types, file_hash, source_type, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (filename, file_path, total_rows, headers, 
                      json.dumps(column_types or {}), file_hash, source_type, 
                      json.dumps(metadata or {})))
                
                doc_id = cur.fetchone()[0]
                self.conn.commit()
                return doc_id
                
        except Exception as e:
            print(f"❌ CSV document insert failed: {e}")
            self.conn.rollback()
            raise

    def insert_csv_rows_batch(self, rows_data: List[Dict]):
        """Batch insert CSV rows with embeddings and proper error handling"""
        if not rows_data:
            return
            
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cur:
                psycopg2.extras.execute_batch(cur, """
                    INSERT INTO csv_rows (csv_doc_id, row_index, row_data, row_text, embedding)
                    VALUES (%(doc_id)s, %(row_index)s, %(row_data)s, %(row_text)s, %(embedding)s::vector);
                """, rows_data)
                self.conn.commit()
                
        except Exception as e:
            print(f"❌ CSV rows batch insert failed: {e}")
            self.conn.rollback()
            raise

    def insert_csv_columns(self, columns_data: List[Dict]):
        """Insert column metadata"""
        if not columns_data:
            return
            
        self._ensure_connection()
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, """
                INSERT INTO csv_columns (csv_doc_id, column_name, column_type, sample_values, 
                                       column_description, embedding)
                VALUES (%(doc_id)s, %(column_name)s, %(column_type)s, %(sample_values)s, 
                        %(column_description)s, %(embedding)s::vector);
            """, columns_data)
            self.conn.commit()

    def insert_table_extraction(self, source_pdf: str, page_number: int, table_name: str,
                               extraction_method: str, csv_content: str, 
                               csv_doc_id: int = None, confidence_score: float = None) -> int:
        """Insert table extraction record with proper error handling"""
        self._ensure_connection()
        
        try:
            with self.conn.cursor() as cur:
                # Validate csv_doc_id if provided
                if csv_doc_id is not None:
                    cur.execute("SELECT id FROM csv_documents WHERE id = %s;", (csv_doc_id,))
                    if not cur.fetchone():
                        print(f"⚠️  Warning: csv_doc_id {csv_doc_id} not found, setting to NULL")
                        csv_doc_id = None
                
                cur.execute("""
                    INSERT INTO table_extractions (source_pdf, page_number, table_name, 
                                                 extraction_method, csv_content, csv_doc_id, confidence_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
                """, (source_pdf, page_number, table_name, extraction_method, 
                      csv_content, csv_doc_id, confidence_score))
                
                extraction_id = cur.fetchone()[0]
                self.conn.commit()
                return extraction_id
                
        except Exception as e:
            print(f"❌ Table extraction insert failed: {e}")
            self.conn.rollback()
            # Return a dummy ID to continue processing
            return -1

    # Search methods
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Original search method for document pages"""
        self._ensure_connection()
        query_vector_str = str(query_vector.tolist())
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id, source_pdf, page_number, html_content, json_metadata, page_summary,
                       1 - (embedding <=> %s::vector) AS similarity 
                FROM document_pages 
                ORDER BY embedding <=> %s::vector 
                LIMIT %s;
                """,
                (query_vector_str, query_vector_str, k)
            )
            results = [dict(row) for row in cur.fetchall()]
        return results

    def search_csv_data(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search CSV data using vector similarity"""
        self._ensure_connection()
        query_vector_str = str(query_vector.tolist())
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT 
                    cr.id, cr.row_index, cr.row_data, cr.row_text,
                    cd.filename, cd.column_names, cd.source_type,
                    1 - (cr.embedding <=> %s::vector) AS similarity
                FROM csv_rows cr
                JOIN csv_documents cd ON cr.csv_doc_id = cd.id
                ORDER BY cr.embedding <=> %s::vector
                LIMIT %s;
            """, (query_vector_str, query_vector_str, k))
            
            return [dict(row) for row in cur.fetchall()]

    def search_by_column_value(self, column_name: str, value: str, similarity_threshold: float = 0.8) -> List[Dict]:
        """Search for specific column values"""
        self._ensure_connection()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT cr.row_data, cd.filename, cd.column_names, cr.row_index
                FROM csv_rows cr
                JOIN csv_documents cd ON cr.csv_doc_id = cd.id
                WHERE cr.row_data->>%s ILIKE %s
                ORDER BY cr.row_index;
            """, (column_name, f"%{value}%"))
            
            return [dict(row) for row in cur.fetchall()]

    def search_combined(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """Search both PDF content and CSV data"""
        self._ensure_connection()
        query_vector_str = str(query_vector.tolist())
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                (SELECT 'pdf' as source_type, id, source_pdf as filename, page_number, 
                        page_summary as content, NULL as row_data, NULL as column_names,
                        1 - (embedding <=> %s::vector) AS similarity
                 FROM document_pages
                 ORDER BY embedding <=> %s::vector
                 LIMIT %s)
                UNION ALL
                (SELECT 'csv' as source_type, cr.id, cd.filename, cr.row_index as page_number,
                        cr.row_text as content, cr.row_data, cd.column_names,
                        1 - (cr.embedding <=> %s::vector) AS similarity
                 FROM csv_rows cr
                 JOIN csv_documents cd ON cr.csv_doc_id = cd.id
                 ORDER BY cr.embedding <=> %s::vector
                 LIMIT %s)
                ORDER BY similarity DESC
                LIMIT %s;
            """, (query_vector_str, query_vector_str, k//2, 
                  query_vector_str, query_vector_str, k//2, k))
            
            return [dict(row) for row in cur.fetchall()]

    def get_csv_document_info(self, filename: str) -> Optional[Dict]:
        """Get CSV document information"""
        self._ensure_connection()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, filename, total_rows, column_names, column_types, source_type, metadata
                FROM csv_documents WHERE filename = %s;
            """, (filename,))
            
            result = cur.fetchone()
            return dict(result) if result else None

    def get_csv_documents_list(self) -> List[Dict]:
        """Get list of all CSV documents"""
        self._ensure_connection()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, filename, total_rows, column_names, source_type, upload_date
                FROM csv_documents ORDER BY upload_date DESC;
            """)
            
            return [dict(row) for row in cur.fetchall()]

    def get_table_extractions(self, source_pdf: str = None) -> List[Dict]:
        """Get table extraction records"""
        self._ensure_connection()
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if source_pdf:
                cur.execute("""
                    SELECT te.*, cd.filename as csv_filename
                    FROM table_extractions te
                    LEFT JOIN csv_documents cd ON te.csv_doc_id = cd.id
                    WHERE te.source_pdf = %s
                    ORDER BY te.page_number, te.id;
                """, (source_pdf,))
            else:
                cur.execute("""
                    SELECT te.*, cd.filename as csv_filename
                    FROM table_extractions te
                    LEFT JOIN csv_documents cd ON te.csv_doc_id = cd.id
                    ORDER BY te.created_at DESC;
                """)
            
            return [dict(row) for row in cur.fetchall()]

    def get_page_images(self, page_id: int) -> List[Dict]:
        """Get images for a specific page"""
        self._ensure_connection()
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT id, image_index, image_data, image_format, image_type, 
                       width, height, position_x, position_y, ai_description
                FROM document_images 
                WHERE page_id = %s 
                ORDER BY image_index;
                """,
                (page_id,)
            )
            results = [dict(row) for row in cur.fetchall()]
        return results

    def optimize_indexes(self):
        """Optimize database indexes for better performance"""
        self._ensure_connection()
        with self.conn.cursor() as cur:
            # Additional performance indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS csv_rows_similarity_idx 
                ON csv_rows USING hnsw (embedding vector_cosine_ops);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS csv_docs_hash_idx 
                ON csv_documents(file_hash);
            """)
            
            # Analyze tables for better query planning
            cur.execute("ANALYZE csv_documents;")
            cur.execute("ANALYZE csv_rows;")
            cur.execute("ANALYZE document_pages;")
            
            self.conn.commit()
        print("✅ Database indexes optimized.")