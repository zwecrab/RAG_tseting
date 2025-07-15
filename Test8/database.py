import psycopg2
import psycopg2.extras
import numpy as np
import hashlib
from typing import List, Dict

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
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
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
            
            cur.execute("CREATE INDEX IF NOT EXISTS doc_page_embedding_idx ON document_pages USING hnsw (embedding vector_l2_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS doc_images_hash_idx ON document_images (image_hash);")
            
            self.conn.commit()
        print("✅ Database setup complete.")

    def clear_documents(self, source_pdf: str):
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM document_images WHERE source_pdf = %s;", (source_pdf,))
            cur.execute("DELETE FROM document_pages WHERE source_pdf = %s;", (source_pdf,))
            deleted_count = cur.rowcount
            self.conn.commit()
        print(f"✅ Cleared existing documents for '{source_pdf}'")

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

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
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

    def get_page_images(self, page_id: int) -> List[Dict]:
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