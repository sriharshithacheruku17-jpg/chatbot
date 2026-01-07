import streamlit as st
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import pyttsx3

load_dotenv()

st.title("ASKMATE")
st.markdown("Upload. Ask. Understand.")

with st.sidebar:
    
   st.markdown("## 🧠 ASKMATE ")

   st.markdown(""" - Your AI buddy for documents""")
   st.markdown(" - Your document stays private and is processed locally")
   
   st.write("")  
   st.write("")

   st.markdown("### 📂 Supported Files")
   st.markdown(""" 
    - 📄 PDF  
    - 📝 DOCX  
    - 📃 TXT  
    - 🌐 HTML   
    - 📊 CSV""")
   
   st.write("")
   st.write("")
   
   st.markdown("### ⚡ How It Works")
   st.markdown("""
    - Upload a document  
    - AI reads & understands it  
    - Ask questions  
    - Get instant answers  
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

uploaded_file = st.file_uploader(" ", type=["pdf", "docx", "txt", "html", "csv"])

if uploaded_file is not None and st.session_state.vectorstore is None:
    with st.spinner("Processing document... This may take a minute."):
        
        file_extension = uploaded_file.name.split('.')[-1] # Saving file temporarily
        temp_filename = f"temp_file.{file_extension}"

        with open(temp_filename, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        loader = UnstructuredFileLoader(temp_filename) 
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Split the text into chunks
        texts = text_splitter.split_documents(documents)
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings) #creating vector store
    
    # Initialize LLM
    try:
        with st.spinner("🧠 Loading AI model ..."):
         llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-base",
            task="text2text-generation",
            device=-1,
            pipeline_kwargs={"max_new_tokens": 150, "temperature": 0}
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

#input
if st.session_state.qa_chain is not None:
    query = st.chat_input("Ask me anything from the document...")
    if query:
        
        st.session_state.chat_history.append({"role": "user", "content": query}) #adds our query to history
    
        with st.spinner("Thinking..."):
            answer = st.session_state.qa_chain.run(query)
        
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        st.rerun()
else:
    st.info("Upload a document to start chatting.")
