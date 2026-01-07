ASKMATE - AI Document Chatbot

ASKMATE is a beginner-friendly AI chatbot that lets you upload documents (like PDFs, Word files, or text) and ask questions about them. It uses smart AI to understand your documents and give accurate answers. No internet needed for answers – everything runs on your computer!

How It Works 
- Upload: Choose a document from your computer.
- Process: AI reads the file, breaks it into chunks, and creates a "smart map" (embeddings) to understand it.
- Ask: Type or speak a question.
- Answer: AI searches the document and gives a reply based on the content.
Chat: Keep asking – it remembers the conversation!

Tech Behind It:
- RAG (Retrieval-Augmented Generation): pulls relevant info in your doc, then generates answers.
- AI Models: Free models like Flan-T5 for understanding and answering.
- Tools: Streamlit for the web app, LangChain for AI chaining, FAISS for fast search.

Requirements
- Computer: Mac, Windows, or Linux.
- Python: Version 3.9 or higher (comes with most computers).
- Internet: Only for first-time model download (~1GB).
- No Cost: Everything is free!

How to Run ASKMATE
Step 1: Install Python if needed
Step 2: Download the Project from github
- Go to GitHub 
- Click "Code" 
- open a folder, e.g., `Desktop/chatbot`.
Step 3: Set Up the Environment
1. Open terminal (on Mac: Spotlight > Terminal).
2. Go to project folder: `cd Desktop/chatbot`
3. Create virtual environment: `python -m venv .venv`
4. Activate it: `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows).
5. Install packages: `pip install -r requirements.txt`

Step 4: Run the App
- In terminal: `streamlit run app.py`
- Open the link in your browser (usually http://localhost:8501).
- Upload a document and start chatting!

Step 5: Use It
- Upload: Click "Browse files" and pick a PDF/TXT/etc.
- Answer: Read the reply or click "🔊 Speak" to hear it.

Features in Detail
- Sidebar: Instructions, file types, how-to guide.
- Voice Output: Click "🔊 Speak" on answers to hear them.
- Chat History: All Q&A saved in the session.
- Error Handling: Shows messages if something goes wrong.


- Streamlit: [docs.streamlit.io](https://docs.streamlit.io) – For building web apps.
- LangChain: [python.langchain.com](https://python.langchain.com) – For AI chains.
- Hugging Face: [huggingface.co](https://huggingface.co) – Free AI models.