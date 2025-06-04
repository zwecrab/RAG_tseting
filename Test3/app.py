import jpype
jvm_path = r'C:\Program Files\Java\jdk-24\bin\server\jvm.dll'
jpype.startJVM(jvm_path)

import os
os.environ['PYTHONIOENCODING'] = 'latin1'

import gradio as gr
from pdf_processor import extract_text_from_pdf, extract_images_from_pdf, extract_tables_from_pdf
from image_captioning import generate_caption
from table_processor import table_to_text
from rag_pipeline import build_index, retrieve_passages, generate_answer
import os

# Store processed PDFs: {pdf_name: (index, texts)}
processed_pdfs = {}

def process_pdfs(pdf_files):
    """Process uploaded PDFs and index their content."""
    processed_pdfs.clear()  # Clear previous data
    pdf_names = []
    for pdf_file in pdf_files:
        pdf_path = pdf_file.name
        pdf_name = os.path.basename(pdf_path)
        
        # Extract content
        paragraphs = extract_text_from_pdf(pdf_path)
        images = extract_images_from_pdf(pdf_path)
        tables = extract_tables_from_pdf(pdf_path)
        
        # Process images and tables
        image_captions = [generate_caption(img) for img in images]
        table_texts = [table_to_text(table) for table in tables]
        
        # Combine all documents
        documents = paragraphs + image_captions + table_texts
        
        # Build index
        index, texts = build_index(documents)
        processed_pdfs[pdf_name] = (index, texts)
        pdf_names.append(pdf_name)
    
    return pdf_names

def answer_question(pdf_name, question):
    """Answer a question based on the selected PDF."""
    if not pdf_name or pdf_name not in processed_pdfs:
        return "Please select a processed PDF."
    if not question.strip():
        return "Please enter a question."
    index, texts = processed_pdfs[pdf_name]
    passages = retrieve_passages(index, texts, question, k=3)
    answer = generate_answer(question, passages)
    return answer

# Gradio interface
with gr.Blocks(title="PDF RAG Analyzer") as demo:
    gr.Markdown("# PDF RAG Analyzer")
    gr.Markdown("Upload PDFs, process them, select one, and ask questions about its content.")
    
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
        process_btn = gr.Button("Process PDFs")
    
    pdf_dropdown = gr.Dropdown(label="Select PDF", choices=[])
    question_input = gr.Textbox(label="Ask a Question", placeholder="What is in this PDF?")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_btn = gr.Button("Ask")
    
    # Event handlers
    process_btn.click(
        fn=process_pdfs,
        inputs=pdf_upload,
        outputs=pdf_dropdown
    )
    ask_btn.click(
        fn=answer_question,
        inputs=[pdf_dropdown, question_input],
        outputs=answer_output
    )

demo.launch()