# ASKMATE - AI PDF Chatbot

A beginner-friendly RAG-based chatbot for PDF Q&A, built with Python and Streamlit. Perfect for presentations!

## Features
- Upload documents (PDF, DOCX, TXT, HTML, MD, CSV) and ask questions.
- Continuous chat with history.
- Voice input: Click "🎤 Voice" to speak questions.
- Voice output: Click "🔊 Speak" to hear answers.
- Cool themes: Choose from gradient, dark, or light mode.
- Free AI models (no API costs).
- Simple UI with instructions.

## Demo
Run: `streamlit run app.py`
Upload a PDF, ask questions like "Summarize the main points."

## What is RAG?

RAG combines retrieval and generation:
1. **Retrieval**: Find relevant information from documents.
2. **Generation**: Use a Large Language Model (LLM) to generate answers based on retrieved info.

## Prerequisites

- Python 3.9+
- Internet connection (to download the free LLM model on first run)

## Setup

1. Clone or download this project.
2. Create a virtual environment (already done: `.venv`).
3. Install dependencies: `pip install -r requirements.txt`
4. No API key needed – uses free local LLM.

## How it works

1. Upload a PDF document.
2. The app loads and splits the PDF into text chunks.
3. Creates embeddings (vector representations) for each chunk.
4. Stores them in a FAISS vector database.
5. When you ask a question, it retrieves relevant chunks.
6. Uses a free local LLM (flan-t5-base) to generate an answer based on the retrieved info.

## Running the App

Run: `streamlit run app.py`

Or with full path: `/Users/sriharshitha/Desktop/chatbot/.venv/bin/python -m streamlit run app.py`

Open the URL shown in the terminal (usually http://localhost:8501).

## Usage

- Upload a PDF.
- Chat continuously with the bot – ask multiple questions, and it remembers the conversation.
- Get answers based on the PDF content (free, runs locally).

## Technologies Used

- LangChain: For chaining LLM and retrieval.
- Sentence Transformers: For creating embeddings.
- FAISS: Vector database.
- Hugging Face Transformers & PyTorch: For running free LLMs locally (flan-t5-base).
- Streamlit: Web interface.
- PyPDF: PDF loading.