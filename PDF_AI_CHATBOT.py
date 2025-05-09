import streamlit as st
import os
from dotenv import load_dotenv
from ask_query import ask_query
from data_ingestion import get_pdf_text, get_text_chunks
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Constants
INDEX_DIR = "faiss_index"
UPLOAD_DIR = "uploaded_pdfs"

# Streamlit page configuration
st.set_page_config(page_title="project_name", page_icon=":robot_face:", layout="centered")

# Sidebar: PDF Upload and Processing at left corner
st.sidebar.header("Upload and Process PDF")
sidebar_status = st.sidebar.empty()
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)
process_trigger = False
if uploaded_files:
    st.sidebar.write(f"Selected {len(uploaded_files)} file(s)")
    process_trigger = st.sidebar.button("Process PDFs")

# Main title always visible
st.title("PDF RAG AI ASSISTANT 🤖")

# Handle PDF processing
if uploaded_files and process_trigger:
    # Show status under button immediately
    sidebar_status.info("Starting PDF processing...")
    # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(file_path)

    # Global spinner for extraction and indexing
    with st.spinner("Extracting text and building FAISS index..."):
        raw_text = get_pdf_text(paths)
        chunks = get_text_chunks(raw_text)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        vectorstore.save_local(INDEX_DIR)
    sidebar_status.success("PDFs processed and FAISS index saved!")

# Main area: Chat Interface
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Welcome to PDF RAG AI ChatBot! 🤖 How can I assist you today?"
    }]

chat_placeholder = st.empty()

def display_chat_history():
    with chat_placeholder.container():
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                _, col = st.columns([1, 5])
                with col:
                    st.chat_message("user").write(message["content"])
            else:
                col, _ = st.columns([5, 1])
                with col:
                    st.chat_message("assistant").write(message["content"])

# Display the chat history
display_chat_history()

# User input
user_input = st.chat_input("Please enter your query here!")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_chat_history()
    with st.spinner("Generating response..."):
        assistant_response = ask_query(user_input)
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    display_chat_history()
