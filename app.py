import streamlit as st
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import os
import json
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

st.title("ASKMATE")

st.markdown("Welcome to ASKMATE! Upload any document (PDF, DOCX, TXT, HTML, MD, CSV) and chat with an AI that answers based on its content. Built with RAG technology.")

# Sidebar with instructions
with st.sidebar:
    st.header("How to Use ASKMATE")
    st.markdown("""
    1. **Upload a Document**: Choose PDF, DOCX, TXT, HTML, MD, or CSV files.
    2. **Wait for Processing**: The AI will analyze it (first time takes longer).
    3. **Ask Questions**: Type in the chat box below.
    4. **Get Answers**: Based on the document content!
    5. **Voice Output**: Click "🔊 Speak" on answers to hear them.
    
    **Tech Used**: RAG (Retrieval + Generation) with free AI models.
    """)
    st.markdown("---")
    st.markdown("**Sample Files**: [PDF](https://www.africau.edu/images/default/sample.pdf), [TXT](https://www.gutenberg.org/files/1342/1342-0.txt), [CSV](https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html) (or use your own).")
    #st.markdown("**Note**: First run downloads the AI model (~1GB). Be patient!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT, HTML, MD, CSV)", type=["pdf", "docx", "txt", "html", "md", "csv"])

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Processing document... This may take a minute."):
        # Save the uploaded file temporarily with correct extension
        file_extension = uploaded_file.name.split('.')[-1]
        temp_filename = f"temp_file.{file_extension}"
        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the document (supports multiple formats)
        loader = UnstructuredFileLoader(temp_filename)
        documents = loader.load()
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Initialize LLM
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base",
            task="text2text-generation",
            device=-1,
            pipeline_kwargs={"max_new_tokens": 100, "temperature": 0}
        )
        st.success("AI Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load AI model: {e}. Check internet and try again.")
        st.stop()
    
    # Create QA chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=st.session_state.vectorstore.as_retriever())
    
    st.success("PDF processed! You can now chat.")

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            if st.button("🔊 Speak", key=f"speak_{i}"):
                engine = pyttsx3.init()
                engine.say(message["content"])
                engine.runAndWait()

# Chat input
if st.session_state.qa_chain is not None:
    query = st.chat_input("Ask a question about the PDF:")
    if query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Generate answer
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(query)
        
        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Rerun to update display
        st.rerun()
else:
    st.info("Please upload a PDF to start chatting.")
