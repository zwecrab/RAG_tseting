# database.py

import psycopg2
import psycopg2.extras
# The pgvector.psycopg2 import is no longer needed with this method
# from pgvector.psycopg2 import register_vector 
import numpy as np
from typing import List, Dict

class VectorStore:
    """
    Handles all database operations using a single, persistent connection
    to a PostgreSQL database with the pgvector extension.
    This version manually casts vectors to bypass library conflicts.
    """
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.conn = None

    def connect(self):
        """
        Establishes the persistent connection to the database.
        """
        if self.conn and not self.conn.closed:
            return

        try:
            print("--- Establishing persistent database connection... ---")
            self.conn = psycopg2.connect(**self.db_config)
            # We no longer call register_vector here.
            print("✅ Persistent database connection established.")
        except psycopg2.OperationalError as e:
            print(f"❌ Could not connect to the database. Please check config.py and ensure the server is running.")
            print(f"   Error details: {e}")
            raise

    def close(self):
        """Closes the persistent database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()
            print("Database connection closed.")

    def _ensure_connection(self):
        """Checks if the connection is active, and reconnects if not."""
        if self.conn is None or self.conn.closed:
            print("Connection lost. Reconnecting...")
            self.connect()

    def setup_database(self):
        """
        Ensures the pgvector extension is enabled and the documents table exists.
        """
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source_pdf TEXT NOT NULL,
                    page_number INT NOT NULL,
                    chunk_type VARCHAR(50),
                    content_text TEXT,
                    embedding VECTOR(384)
                );
            """)
            # Optional: Create an index for faster searching
            # You only need to run this once manually or include it here.
            cur.execute("CREATE INDEX IF NOT EXISTS doc_embedding_idx ON documents USING hnsw (embedding vector_l2_ops);")
            self.conn.commit()
        print("✅ Database setup complete.")

    def insert_chunks(self, chunks: List[Dict]):
        """
        Inserts a list of processed text chunks with their embeddings into the database.
        """
        self._ensure_connection()
        
        # Prepare data for insertion
        data_to_insert = []
        for chunk in chunks:
            # Convert numpy array to string format that pgvector understands: '[1,2,3]'
            embedding_str = str(chunk['embedding'].tolist())
            data_to_insert.append({
                'source_pdf': chunk['source_pdf'],
                'page_number': chunk['page_number'],
                'chunk_type': chunk['chunk_type'],
                'content_text': chunk['content_text'],
                'embedding': embedding_str # Use the string representation
            })

        with self.conn.cursor() as cur:
            # The %s for embedding will now be a string, which PostgreSQL will cast to VECTOR
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO documents (source_pdf, page_number, chunk_type, content_text, embedding)
                VALUES (%(source_pdf)s, %(page_number)s, %(chunk_type)s, %(content_text)s, %(embedding)s::vector);
                """,
                data_to_insert
            )
            self.conn.commit()
        print(f"Successfully inserted {len(chunks)} chunks into the database.")

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Performs a similarity search in the database.
        """
        self._ensure_connection()
        
        # Convert query vector to string format
        query_vector_str = str(query_vector.tolist())
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Use ::vector to cast the string parameter to a vector for the search
            cur.execute(
                "SELECT id, source_pdf, page_number, content_text, 1 - (embedding <=> %s::vector) AS similarity FROM documents ORDER BY embedding <=> %s::vector LIMIT %s;",
                (query_vector_str, query_vector_str, k)
            )
            results = [dict(row) for row in cur.fetchall()]
        return results