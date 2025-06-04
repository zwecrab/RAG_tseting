# PDF RAG Analyzer

This is a Retrieval-Augmented Generation (RAG) application with a Gradio web UI that analyzes text, images, and tabular data from PDFs. It runs offline on a personal laptop with an RTX 3070 GPU, using lightweight models and no API keys.

## Features
- Upload multiple PDF files.
- Process PDFs to extract text, images, and tables.
- Select a PDF and ask questions about its content.
- Analyzes text, generates image captions, and interprets tables using a RAG pipeline.
- Runs entirely offline with local models.

## Prerequisites
- Python 3.8 or higher
- NVIDIA RTX 3070 GPU (or compatible CUDA-enabled GPU)
- Approximately 5-10 GB of free disk space for models and dependencies

## Setup Instructions

1. **Clone or download the project files** to your local machine.

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the models**:
   Run the provided script to download the necessary models for offline use:
   ```bash
   python download_models.py
   ```

5. **Run the application**:
   ```bash
   python app.py
   ```
   Open the URL provided by Gradio (e.g., `http://127.0.0.1:7860`) in your browser.

## Usage
1. **Upload PDFs**: Use the "Upload PDFs" section to upload one or more PDF files.
2. **Process PDFs**: Click "Process PDFs" to extract and index the content. This may take a few moments depending on the PDF size and number.
3. **Select a PDF**: Choose a PDF from the dropdown menu.
4. **Ask a Question**: Enter a question about the selected PDF in the text box and click "Ask".
5. **View the Answer**: The generated answer will appear in the output box.

## Notes
- The application uses lightweight models (`all-MiniLM-L6-v2`, `flan-t5-small`, `vit-gpt2-image-captioning`) optimized for an RTX 3070 GPU with 8GB VRAM.
- Processing large PDFs or many PDFs at once may consume significant memory and time.
- FAISS uses the CPU version (`faiss-cpu`) for simplicity; switch to `faiss-gpu` if desired and CUDA is configured.

## Dependencies
See `requirements.txt` for the full list of Python packages required.

## Troubleshooting
- **GPU Memory Errors**: Reduce the number of PDFs processed simultaneously or use smaller PDFs.
- **Model Download Issues**: Ensure an internet connection during the initial `download_models.py` run, then it runs offline.
- **Java Errors with tabula-py**: Install a Java Runtime Environment (JRE) if not already present.

Enjoy analyzing your PDFs offline!