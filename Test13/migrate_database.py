#!/usr/bin/env python3
"""
Database Migration Script for Enhanced RAG System
Migrates existing database to support CSV processing and enhanced features
"""

import psycopg2
import psycopg2.extras
import config
import sys
from datetime import datetime

class DatabaseMigrator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        
    def connect(self):
        """Connect to database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            print("✅ Connected to database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
    
    def check_existing_tables(self):
        """Check which tables already exist"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            
        print(f"📊 Existing tables: {', '.join(existing_tables)}")
        return existing_tables
    
    def backup_existing_data(self):
        """Create backup of existing data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_commands = []
        
        with self.conn.cursor() as cur:
            # Check if original tables exist
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'document_pages'
                );
            """)
            
            if cur.fetchone()[0]:
                backup_commands.append(f"CREATE TABLE document_pages_backup_{timestamp} AS SELECT * FROM document_pages;")
                backup_commands.append(f"CREATE TABLE document_images_backup_{timestamp} AS SELECT * FROM document_images;")
                
                for cmd in backup_commands:
                    cur.execute(cmd)
                    print(f"✅ Executed: {cmd}")
        
        self.conn.commit()
        print(f"📁 Backup created with timestamp: {timestamp}")
        return timestamp
    
    def enable_pgvector(self):
        """Enable pgvector extension"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.conn.commit()
                print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"⚠️  pgvector extension error: {e}")
            print("Please ensure pgvector is installed on your PostgreSQL server")
    
    def create_csv_tables(self):
        """Create new CSV-specific tables"""
        csv_tables = [
            """
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
                source_type VARCHAR(20) DEFAULT 'upload'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS csv_rows (
                id SERIAL PRIMARY KEY,
                csv_doc_id INTEGER REFERENCES csv_documents(id) ON DELETE CASCADE,
                row_index INTEGER NOT NULL,
                row_data JSONB,
                row_text TEXT,
                embedding VECTOR(384),
                created_at TIMESTAMP DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS csv_columns (
                id SERIAL PRIMARY KEY,
                csv_doc_id INTEGER REFERENCES csv_documents(id) ON DELETE CASCADE,
                column_name TEXT NOT NULL,
                column_type TEXT,
                sample_values TEXT[],
                column_description TEXT,
                embedding VECTOR(384)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS table_extractions (
                id SERIAL PRIMARY KEY,
                source_pdf TEXT NOT NULL,
                page_number INT NOT NULL,
                table_name TEXT,
                extraction_method VARCHAR(50),
                csv_content TEXT,
                csv_doc_id INTEGER REFERENCES csv_documents(id),
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        ]
        
        with self.conn.cursor() as cur:
            for table_sql in csv_tables:
                cur.execute(table_sql)
                print(f"✅ Created/verified table")
        
        self.conn.commit()
        print("📊 CSV tables created successfully")
    
    def create_enhanced_indexes(self):
        """Create enhanced indexes for better performance"""
        indexes = [
            # Original indexes
            "CREATE INDEX IF NOT EXISTS doc_page_embedding_idx ON document_pages USING hnsw (embedding vector_l2_ops);",
            "CREATE INDEX IF NOT EXISTS doc_images_hash_idx ON document_images (image_hash);",
            
            # New CSV indexes
            "CREATE INDEX IF NOT EXISTS csv_rows_embedding_idx ON csv_rows USING hnsw (embedding vector_l2_ops);",
            "CREATE INDEX IF NOT EXISTS csv_columns_embedding_idx ON csv_columns USING hnsw (embedding vector_l2_ops);",
            "CREATE INDEX IF NOT EXISTS csv_rows_doc_idx ON csv_rows(csv_doc_id);",
            "CREATE INDEX IF NOT EXISTS csv_rows_data_gin_idx ON csv_rows USING gin (row_data);",
            "CREATE INDEX IF NOT EXISTS csv_docs_filename_idx ON csv_documents(filename);",
            "CREATE INDEX IF NOT EXISTS csv_docs_hash_idx ON csv_documents(file_hash);",
            "CREATE INDEX IF NOT EXISTS table_extractions_pdf_idx ON table_extractions(source_pdf, page_number);",
            
            # Performance indexes
            "CREATE INDEX IF NOT EXISTS csv_rows_similarity_idx ON csv_rows USING hnsw (embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS doc_pages_source_idx ON document_pages(source_pdf);",
            "CREATE INDEX IF NOT EXISTS doc_pages_created_idx ON document_pages(created_at);",
            "CREATE INDEX IF NOT EXISTS csv_docs_created_idx ON csv_documents(upload_date);",
        ]
        
        with self.conn.cursor() as cur:
            for index_sql in indexes:
                try:
                    cur.execute(index_sql)
                    print(f"✅ Created index")
                except Exception as e:
                    print(f"⚠️  Index creation warning: {e}")
        
        self.conn.commit()
        print("📈 Enhanced indexes created")
    
    def update_existing_tables(self):
        """Update existing tables with new columns if needed"""
        updates = [
            # Add any new columns to existing tables
            """
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'document_pages' AND column_name = 'processing_version'
                ) THEN
                    ALTER TABLE document_pages ADD COLUMN processing_version INTEGER DEFAULT 1;
                END IF;
            END $$;
            """,
            """
            DO $$ 
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = 'document_images' AND column_name = 'extraction_method'
                ) THEN
                    ALTER TABLE document_images ADD COLUMN extraction_method VARCHAR(50) DEFAULT 'legacy';
                END IF;
            END $$;
            """
        ]
        
        with self.conn.cursor() as cur:
            for update_sql in updates:
                cur.execute(update_sql)
                print("✅ Updated existing table structure")
        
        self.conn.commit()
        print("🔄 Existing tables updated")
    
    def create_migration_log(self, backup_timestamp):
        """Create migration log table and record migration"""
        with self.conn.cursor() as cur:
            # Create migration log table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS migration_log (
                    id SERIAL PRIMARY KEY,
                    migration_date TIMESTAMP DEFAULT NOW(),
                    migration_type VARCHAR(50),
                    backup_timestamp VARCHAR(20),
                    success BOOLEAN,
                    notes TEXT
                );
            """)
            
            # Record this migration
            cur.execute("""
                INSERT INTO migration_log (migration_type, backup_timestamp, success, notes)
                VALUES (%s, %s, %s, %s);
            """, (
                "enhanced_csv_support",
                backup_timestamp,
                True,
                "Enhanced system with CSV processing, pgvector, and improved table extraction"
            ))
        
        self.conn.commit()
        print("📝 Migration logged")
    
    def verify_migration(self):
        """Verify migration was successful"""
        required_tables = [
            'document_pages', 'document_images',
            'csv_documents', 'csv_rows', 'csv_columns', 'table_extractions'
        ]
        
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = ANY(%s);
            """, (required_tables,))
            
            existing_tables = [row[0] for row in cur.fetchall()]
            
            missing_tables = set(required_tables) - set(existing_tables)
            
            if missing_tables:
                print(f"❌ Missing tables: {', '.join(missing_tables)}")
                return False
            
            # Check pgvector extension
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            if not cur.fetchone()[0]:
                print("❌ pgvector extension not found")
                return False
            
            # Check for vector columns
            cur.execute("""
                SELECT column_name, table_name FROM information_schema.columns 
                WHERE data_type = 'USER-DEFINED' AND udt_name = 'vector';
            """)
            
            vector_columns = cur.fetchall()
            if len(vector_columns) < 3:  # Should have at least 3 vector columns
                print(f"⚠️  Only {len(vector_columns)} vector columns found")
            
            print(f"✅ Vector columns: {', '.join([f'{col[1]}.{col[0]}' for col in vector_columns])}")
        
        print("✅ Migration verification successful")
        return True
    
    def run_migration(self):
        """Run complete migration process"""
        print("🚀 Starting Enhanced RAG System Migration")
        print("=" * 50)
        
        # Connect to database
        self.connect()
        
        # Check existing state
        existing_tables = self.check_existing_tables()
        
        # Backup existing data
        backup_timestamp = self.backup_existing_data()
        
        try:
            # Enable pgvector
            self.enable_pgvector()
            
            # Create new CSV tables
            self.create_csv_tables()
            
            # Update existing tables
            self.update_existing_tables()
            
            # Create enhanced indexes
            self.create_enhanced_indexes()
            
            # Create migration log
            self.create_migration_log(backup_timestamp)
            
            # Verify migration
            if self.verify_migration():
                print("\n" + "=" * 50)
                print("✅ Migration completed successfully!")
                print("🎉 Your database now supports:")
                print("   - CSV file processing and storage")
                print("   - Enhanced table extraction")
                print("   - pgvector for similarity search")
                print("   - Improved performance indexes")
                print("   - Table extraction logging")
                print(f"   - Data backup: {backup_timestamp}")
                print("\nYou can now use the enhanced RAG system!")
            else:
                print("\n❌ Migration verification failed")
                print("Please check the errors above and try again")
                
        except Exception as e:
            print(f"\n❌ Migration failed: {e}")
            print("Your original data has been backed up.")
            print("Please check the error and try again.")
            import traceback
            traceback.print_exc()
        
        finally:
            self.close()

def main():
    """Main migration function"""
    print("Enhanced RAG System - Database Migration")
    print("This will upgrade your database to support CSV processing")
    print("and enhanced table extraction capabilities.")
    
    # Confirm migration
    response = input("\nDo you want to proceed with the migration? (y/N): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        return
    
    # Run migration
    migrator = DatabaseMigrator(config.DB_CONFIG)
    migrator.run_migration()

if __name__ == "__main__":
    main()