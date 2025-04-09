import streamlit as st
import os
import time
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
)

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Chat with Websites & Documents", page_icon="🌐", layout="wide")

# App title and description
st.title("🌐 Chat with Websites & Documents")
st.markdown(
    """
    Enter a URL to chat with a website or upload documents to ask questions about their content.
    This app uses Selenium for JavaScript-enabled websites, document loaders for various file formats, and Gemini to understand the content.
"""
)

# Get API key from environment variable
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error(
        "Google API key not found in environment variables. Please create a .env file with GOOGLE_API_KEY=your_api_key"
    )

# Dictionary mapping file extensions to appropriate loaders
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
}

# Function to load website content using Selenium
def load_website_with_selenium(url, wait_time=5):
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        driver.get(url)

        # Wait for JavaScript to load
        time.sleep(wait_time)

        # Get the page source after JavaScript execution
        page_source = driver.page_source

        # Parse with BeautifulSoup to extract text
        soup = BeautifulSoup(page_source, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Get text
        text = soup.get_text()

        # Close the driver
        driver.quit()

        # Create a Document object
        doc = Document(page_content=text, metadata={"source": url})
        return [doc]

    except Exception as e:
        st.error(f"Error loading website with Selenium: {str(e)}")
        return []

# Function to load documents based on file extension
def load_document(file):
    file_extension = os.path.splitext(file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        if file_extension in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_extension]
            loader = loader_class(temp_file_path)
            documents = loader.load()
            return documents
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []
    except Exception as e:
        st.error(f"Error loading document: {str(e)}")
        return []
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "url" not in st.session_state:
    st.session_state.url = ""
if "document_name" not in st.session_state:
    st.session_state.document_name = ""
if "source_type" not in st.session_state:
    st.session_state.source_type = None

# Sidebar for input selection and processing
with st.sidebar:
    st.header("Setup")

    # Display API key status
    if google_api_key:
        st.success("Google API Key found in .env file")
    else:
        st.error(
            "Google API Key not found. Create a .env file with GOOGLE_API_KEY=your_api_key"
        )
    
    # Source selection
    source_type = st.radio("Select Source Type:", ["Website", "Document"])
    st.session_state.source_type = source_type
    
    if source_type == "Website":
        # Website URL input section
        url = st.text_input("Enter Website URL:", placeholder="https://example.com")
        wait_time = st.slider(
            "JavaScript Load Wait Time (seconds):",
            1,
            15,
            5,
            help="How long to wait for JavaScript to load on the page",
        )

        if st.button("Process Website"):
            if google_api_key and url:
                st.session_state.url = url
                st.session_state.document_name = ""  # Clear document name

                with st.spinner("Processing website content (this may take a minute)..."):
                    try:
                        # Load website content using Selenium
                        documents = load_website_with_selenium(url, wait_time)

                        if not documents:
                            st.error("Could not extract content from the website.")
                        else:
                            # Split documents into chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            )
                            chunks = text_splitter.split_documents(documents)

                            # Create embeddings and index
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", google_api_key=google_api_key
                            )
                            vectorstore = FAISS.from_documents(chunks, embeddings)

                            # Create conversation chain
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )

                            st.session_state.conversation = (
                                ConversationalRetrievalChain.from_llm(
                                    llm=ChatGoogleGenerativeAI(
                                        model="gemini-1.5-flash",
                                        temperature=0.7,
                                        google_api_key=google_api_key,
                                    ),
                                    retriever=vectorstore.as_retriever(
                                        search_kwargs={"k": 5}
                                    ),
                                    memory=memory,
                                )
                            )

                            st.success(
                                f"Website processed successfully! You can now chat about {url}"
                            )

                            # Clear previous chat
                            st.session_state.chat_history = []
                            st.session_state.messages = []
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                if not google_api_key:
                    st.warning("Google API Key not found in .env file")
                if not url:
                    st.warning("Please enter a URL")
    
    else:  # Document upload section
        st.write("Upload Document(s)")
        uploaded_files = st.file_uploader(
            "Choose file(s)", 
            accept_multiple_files=True,
            type=list(set(ext[1:] for ext in LOADER_MAPPING.keys())),
            help="Supported formats: PDF, DOCX, TXT, CSV, PPTX, HTML"
        )
        
        if st.button("Process Document(s)"):
            if google_api_key and uploaded_files:
                all_documents = []
                file_names = []
                
                with st.spinner("Processing document(s)..."):
                    try:
                        for file in uploaded_files:
                            st.info(f"Processing {file.name}...")
                            documents = load_document(file)
                            if documents:
                                all_documents.extend(documents)
                                file_names.append(file.name)
                        
                        if not all_documents:
                            st.error("Could not extract content from the document(s).")
                        else:
                            document_names = ", ".join(file_names)
                            st.session_state.document_name = document_names
                            st.session_state.url = ""  # Clear URL
                            
                            # Split documents into chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000, chunk_overlap=200
                            )
                            chunks = text_splitter.split_documents(all_documents)
                            
                            # Create embeddings and index
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", google_api_key=google_api_key
                            )
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                            
                            # Create conversation chain
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )
                            
                            st.session_state.conversation = (
                                ConversationalRetrievalChain.from_llm(
                                    llm=ChatGoogleGenerativeAI(
                                        model="gemini-1.5-flash",
                                        temperature=0.7,
                                        google_api_key=google_api_key,
                                    ),
                                    retriever=vectorstore.as_retriever(
                                        search_kwargs={"k": 5}
                                    ),
                                    memory=memory,
                                )
                            )
                            
                            st.success(
                                f"Document(s) processed successfully! You can now chat about {document_names}"
                            )
                            
                            # Clear previous chat
                            st.session_state.chat_history = []
                            st.session_state.messages = []
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                if not google_api_key:
                    st.warning("Google API Key not found in .env file")
                if not uploaded_files:
                    st.warning("Please upload at least one document")

    # Add a section with manual model name overrides
    st.markdown("---")
    st.subheader("Advanced Settings")
    with st.expander("Model Configuration"):
        embedding_model = st.text_input(
            "Embedding Model Name:", value="models/embedding-001"
        )
        llm_model = st.text_input("LLM Model Name:", value="gemini-1.5-flash")
        st.info(
            "If you encounter model name errors, try different formats like 'embedding-001' or 'models/embedding-001' for embeddings, and 'gemini-pro' or 'gemini-1.5-pro' for the LLM."
        )

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        This app allows you to chat with website content and documents using:
        - Streamlit for the UI
        - Selenium for loading JavaScript-dependent websites
        - Document loaders for various file formats (PDF, DOCX, TXT, CSV, PPTX, HTML)
        - Langchain for document processing
        - Google's Gemini for understanding and responding
        - FAISS for efficient vector storage
    """
    )

# Display current source information
if st.session_state.url:
    st.info(f"Currently loaded source: Website - {st.session_state.url}")
elif st.session_state.document_name:
    st.info(f"Currently loaded source: Document(s) - {st.session_state.document_name}")
else:
    source_type = st.session_state.source_type or "Website"
    if source_type == "Website":
        st.warning("No website processed yet. Please enter a URL and process it.")
    else:
        st.warning("No documents processed yet. Please upload documents and process them.")

# Display chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_query = st.chat_input("Ask something about the content...")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    if st.session_state.conversation:
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation.invoke(
                        {"question": user_query}
                    )
                    st.write(response["answer"])

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["answer"]}
                    )
                except Exception as e:
                    error_message = f"Error generating response: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message}
                    )
    else:
        with st.chat_message("assistant"):
            if not st.session_state.url and not st.session_state.document_name:
                source_type = st.session_state.source_type or "Website"
                if source_type == "Website":
                    st.warning("Please enter a URL and process it first.")
                else:
                    st.warning("Please upload documents and process them first.")
            else:
                st.warning("Processing content. Please wait...")
elif not google_api_key:
    st.warning("Google API Key not found in .env file")

# Add a button to clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.conversation:
        st.session_state.conversation.memory.clear()