# Enhanced RAG System Setup Guide

This guide will help you set up the enhanced RAG system with CSV processing and pgvector support.

## 📋 Prerequisites

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. PostgreSQL Database
- PostgreSQL 12 or higher
- **pgvector extension** (REQUIRED for vector operations)

### 3. Java Runtime
- Java 8 or higher (required for tabula-py)

### 4. OpenAI API Access
- Azure OpenAI Service (recommended) OR Standard OpenAI API
- API key with vision capabilities (GPT-4 with vision)

## 🚀 Installation Steps

### Step 1: Install pgvector Extension

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql-14-pgvector
# or for other versions: postgresql-12-pgvector, postgresql-13-pgvector, etc.
```

**For macOS (using Homebrew):**
```bash
brew install pgvector
```

**For Windows or Manual Installation:**
1. Download from: https://github.com/pgvector/pgvector
2. Follow the build instructions in the repository

**Enable in PostgreSQL:**
```sql
-- Connect to your database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install required packages
pip install -r requirements.txt
```

**Create requirements.txt:**
```txt
# Core dependencies
psycopg2-binary>=2.9.0
pandas>=1.5.0
numpy>=1.21.0
pymupdf>=1.23.0
tabula-py>=2.7.0
requests>=2.28.0

# AI/ML dependencies
sentence-transformers>=2.2.0
transformers>=4.21.0
torch>=1.12.0

# Database
psycopg2-binary>=2.9.0

# Optional for better performance
faiss-cpu>=1.7.0  # For faster similarity search
```

### Step 3: Configure Database

1. **Create Database:**
```sql
CREATE DATABASE rag_db;
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;
```

2. **Update config.py:**
```python
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "rag_user",  # Your database user
    "password": "your_password",  # Your database password
    "dbname": "rag_db"  # Your database name
}
```

### Step 4: Configure OpenAI API

**For Azure OpenAI (Recommended):**
```python
# In config.py
AZURE_ENDPOINT = "https://your-resource-name.openai.azure.com"
OPENAI_API_KEY = "your-azure-api-key"
DEPLOYMENT_NAME = "gpt-4o"  # Your deployment name
```

**For Standard OpenAI:**
```python
# In config.py
AZURE_ENDPOINT = None
OPENAI_API_KEY = "sk-your-openai-api-key"
DEPLOYMENT_NAME = "gpt-4o"
```

### Step 5: Configure Java (for tabula-py)

**Windows:**
```python
# In config.py
JAVA_HOME_PATH = "C:/Program Files/Java/jdk-11"
```

**Linux/Mac:**
```python
# In config.py
JAVA_HOME_PATH = "/usr/lib/jvm/java-11-openjdk"
```

### Step 6: Run Migration (for existing installations)

If you have an existing installation:
```bash
python migrate_database.py
```

For new installations, this is handled automatically.

## 🧪 Testing Installation

### Test 1: Database Connection
```python
python -c "
from database import VectorStore
import config
db = VectorStore(config.DB_CONFIG)
db.connect()
print('✅ Database connection successful')
db.close()
"
```

### Test 2: pgvector Extension
```python
python -c "
from database import VectorStore
import config
db = VectorStore(config.DB_CONFIG)
db.connect()
db.setup_database()
print('✅ pgvector setup successful')
db.close()
"
```

### Test 3: OpenAI API
```python
python -c "
from services import ImageToTextService
import config
service = ImageToTextService(config.OPENAI_API_KEY, config.AZURE_ENDPOINT)
print('✅ OpenAI service initialized')
"
```

### Test 4: Complete System
```bash
python main.py
# Select option 5 to see database statistics
```

## 📊 First Usage

### Process Your First PDF
1. Run: `python main.py`
2. Select option 1: "Process a PDF file"
3. Enter path to your PDF file
4. Wait for processing to complete

### Process Your First CSV
1. Run: `python main.py`
2. Select option 2: "Process a CSV file"
3. Enter path to your CSV file
4. Wait for processing to complete

### Ask Your First Question
1. Run: `python main.py`
2. Select option 4: "Ask a question"
3. Enter your question (e.g., "What is the value in Table 25 for Bob won in 2009?")
4. View the results

## 📁 Project Structure

```
enhanced-rag-system/
├── main.py                 # Main application
├── config.py              # Configuration
├── database.py            # Database operations
├── services.py            # AI services
├── processor.py           # Content processing
├── extractor.py           # File extraction
├── migrate_database.py    # Database migration
├── SETUP.md              # This file
├── requirements.txt       # Python dependencies
└── logs/                 # Log files (created automatically)
```

## 🔧 Troubleshooting

### Common Issues

**1. pgvector not found:**
```
ERROR: extension "vector" is not available
```
**Solution:** Install pgvector extension (see Step 1)

**2. Java not found:**
```
ERROR: Java executable not found
```
**Solution:** Install Java and set JAVA_HOME_PATH in config.py

**3. OpenAI API errors:**
```
ERROR: 401 Unauthorized
```
**Solution:** Check your API key and endpoint configuration

**4. Database connection errors:**
```
ERROR: could not connect to server
```
**Solution:** Check PostgreSQL is running and credentials are correct

**5. Memory issues with large files:**
```
ERROR: Memory allocation failed
```
**Solution:** Reduce batch sizes in config.py or process smaller files

### Performance Optimization

**1. Database Performance:**
```sql
-- Run in your database
VACUUM ANALYZE;
REINDEX DATABASE rag_db;
```

**2. Increase Work Memory:**
```sql
-- In postgresql.conf
work_mem = 256MB
shared_buffers = 512MB
```

**3. Python Memory:**
```python
# In config.py
BATCH_SIZE = 50  # Reduce if memory issues
PERFORMANCE_CONFIG["embedding_batch_size"] = 25
```

## 🚨 Important Notes

1. **Backup Your Data:** Always backup your database before migration
2. **API Costs:** Be aware of OpenAI API costs for large files
3. **File Sizes:** Large PDFs may take significant time to process
4. **Concurrent Processing:** Limit concurrent extractions to avoid API limits

## 🎯 Next Steps

1. **Process your documents:** Start with smaller files to test
2. **Optimize configuration:** Adjust batch sizes and thresholds
3. **Monitor performance:** Use database statistics to optimize
4. **Scale up:** Process larger document collections

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check log files in the `logs/` directory
4. Ensure your database and API credentials are correct

## 🔄 Updates

To update the system:
1. Backup your database
2. Update code files
3. Run migration script if needed
4. Test with sample data

Happy processing! 🎉