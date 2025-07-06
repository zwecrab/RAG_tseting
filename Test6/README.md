# PDF Semantic Search Prototype

## 1. Project Overview

This project is a Python-based prototype that demonstrates a Retrieval-Augmented Generation (RAG) pipeline for PDF documents. It transforms complex PDFs—containing a mix of standard text, tables, and images—into a searchable knowledge base. The core functionality allows a user to ask questions in natural language and retrieve the most semantically relevant text snippets from the document library.

This system is designed to move beyond simple keyword matching and understand the *meaning* behind a user's query to find contextually accurate information, even if the exact words aren't present in the text.

---

## 2. How It Works: The Detailed Pipeline

The prototype operates through a multi-stage data processing pipeline. Each step is crucial for transforming unstructured PDF data into a queryable index.

### Step 1: Data Extraction and Chunking
The process begins by ingesting PDF files. Since PDFs are visual formats, simply extracting text isn't enough. Our script intelligently handles different content types:

* **Plain Text:** Standard paragraphs and sentences are extracted. To ensure each vector represents a focused idea, this text is broken down into smaller **chunks** (individual lines or short paragraphs).
* **Tables:** Using `pdfplumber`, the script identifies tables. Instead of extracting the text as a jumbled block, it processes the table row by row. Each row is converted into a descriptive sentence (e.g., `"Table data: details: Total electricity usage costs, unit: baht, 2021: 3,058,617.65"`). This preserves the relationship between the data and its corresponding headers, which is critical for accurate retrieval.
* **Images (OCR):** When images are detected, `pytesseract` is used to perform Optical Character Recognition (OCR), extracting any embedded text. This text is also chunked into smaller lines.

The output of this stage is a clean list of small, semantically-focused text chunks.

### Step 2: Vectorization (Turning Meaning into Math)
This is where the "semantic" understanding comes from. Each text chunk is fed into a `sentence-transformer` model (`all-MiniLM-L6-v2`). The model converts the text into a **vector**—a fixed-size list of numbers (in this case, 384 numbers).

This process works as follows:
1.  **Tokenization:** The text is broken into words or sub-words (tokens).
2.  **Embedding:** Each token is converted into an initial numerical vector.
3.  **Contextualization:** A transformer network analyzes the entire sequence of tokens, adjusting each token's vector based on its neighbors. This allows the model to grasp context (e.g., understanding that "costs" refers to "electricity").
4.  **Pooling:** The vectors for all tokens in the chunk are averaged to produce a single vector that represents the overall meaning of the chunk.

### Step 3: Indexing for Search
The generated vectors need to be stored in a way that allows for efficient searching. This prototype uses **FAISS (Facebook AI Similarity Search)**.

* All vectors are loaded into a FAISS index. FAISS creates a specialized data structure that is highly optimized for finding the "nearest neighbors" to a given query vector in a high-dimensional space.
* This process is incredibly fast, allowing for near-instantaneous search even with millions of vectors.

### Step 4: Semantic Search
When a user provides a query:
1.  The user's query string is passed through the *exact same* vectorization process, creating a query vector.
2.  FAISS takes this query vector and compares it against all the vectors in the index, calculating the distance (similarity) between them.
3.  FAISS returns the IDs of the vectors that are "closest" (most similar) to the query vector.
4.  The system uses these IDs to retrieve the original text chunks associated with those vectors and presents them to the user as the search results.

---

## 3. Models & Tools Used

* **Language**: **Python 3**
* **PDF Parsing**: **`pdfplumber`** - A powerful tool for extracting text and tables while preserving structure.
* **OCR Engine**: **`pytesseract`** - A Python wrapper for Google's Tesseract-OCR engine, used to extract text from images.
* **Embedding Model**: **`sentence-transformers/all-MiniLM-L6-v2`** - A high-performance model that balances speed and accuracy, converting text chunks into 384-dimensional vectors.
* **Vector Index**: **`faiss-cpu`** - A library from Facebook AI for highly efficient similarity search of vectors on the CPU.
* **External Programs**:
    * **Tesseract-OCR**: The underlying OCR engine.
    * **ImageMagick**: An image processing suite used by `pdfplumber` to handle images.
    * **Ghostscript**: A PDF interpreter required by ImageMagick.

---

## 4. Database & Storage Explained

In this prototype, the "database" consists of two local files that work together:

1.  **`text_data.pkl` (The "What")**: This file stores the human-readable content. It's a Python pickle file containing a list of all the text chunks extracted from the PDFs. Each entry also contains metadata, like the source PDF filename.

2.  **`faiss_index.bin` (The "Where")**: This file stores the numerical vectors. It's a binary file containing the FAISS index structure. The vectors are stored in the **exact same order** as the text chunks in `text_data.pkl`.

This parallel structure allows the system to link the two: FAISS finds the *position* (e.g., index #42) of the most similar vector, and the program then retrieves the text chunk at position #42 from the `text_data.pkl` list.

### Transitioning to `pgvector`
In a production system, these files would be replaced by a PostgreSQL database with the `pgvector` extension. The storage concept remains the same, but it becomes much more robust. You would have a single table with columns like:

| id (PK) | source_pdf (TEXT) | content_text (TEXT) | embedding (VECTOR(384)) |
| :--- | :--- | :--- | :--- |
| 1 | doc1.pdf | Chunk 1 text... | [0.12, -0.54, ...] |
| 2 | doc1.pdf | Chunk 2 text... | [0.88, 0.21, ...] |

Here, the text and its corresponding vector are stored together in the same row, which is a more scalable and manageable approach.

---

## 5. Step-by-Step Workflow

1.  **Prerequisites**: Ensure Python, Tesseract-OCR, ImageMagick, and Ghostscript are installed on your system.
2.  **Setup Environment**:
    ```bash
    # Create and activate a Python virtual environment
    python -m venv venv
    source venv/bin/activate

    # Install all required Python packages
    pip install -r requirements.txt
    ```
3.  **Add Documents**: Place your PDF files into the `/pdfs` directory.
4.  **Build the Index**: Run the indexer to process all PDFs.
    ```bash
    python main.py index
    ```
5.  **Perform a Search**: Query the index with a natural language question.
    ```bash
    python main.py search "what is the total volume of waste in 2021?"
    ```

---

## 6. Evaluation Results

The prototype was evaluated using an 8-question test set to measure its retrieval accuracy.

* **Metric Used**: **Mean Reciprocal Rank (MRR)**. This metric evaluates how well the system ranks the correct answer. A higher score is better.
* **Final Score**: **0.4271**

### Breakdown of Results:

| Question ID | Query | Rank of Correct Answer |
| :--- | :--- | :--- |
| Q1 | Electricity consumption 2021? | 6 |
| Q2 | Electricity spend 2565? | 1 |
| Q3 | Diesel consumption 2566? | 1 |
| Q4 | Waste volume 2021? | 1 |
| Q5 | Total employees 2566? | Not Found |
| Q6 | Employee training costs 2566? | 4 |
| Q7 | Work injuries 2021? | Not Found |
| Q8 | Female employees 2565? | Not Found |

### Analysis:
An MRR score of **0.4271** is a promising start for a prototype. It indicates that the system is frequently able to find the correct information within the top results.

The queries that **failed** (Q5, Q7, Q8) highlight key areas for improvement:
* **Chunking Ambiguity**: The text chunks for "total employees," "work-related injuries," and "female employees" might be too similar to other, nearby text (like data for male employees or different years), confusing the model.
* **Table Complexity**: The tables containing this data might have complex layouts (merged cells, sub-headers) that the current table processing logic doesn't fully parse, leading to less clean, less distinct chunks.

Future work should focus on more advanced chunking strategies to better isolate these specific data points.
