import streamlit as st
import os
import time
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

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🌐", layout="wide")

# App title and description
st.title("🌐 Chat with JavaScript-Enabled Websites")
st.markdown(
    """
    Enter a URL and ask questions about the content of the website.
    This app uses Selenium to load JavaScript-dependent websites and Gemini to understand the content.
"""
)

# Get API key from environment variable
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error(
        "Google API key not found in environment variables. Please create a .env file with GOOGLE_API_KEY=your_api_key"
    )


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


# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "url" not in st.session_state:
    st.session_state.url = ""

# Sidebar for URL input and website processing
with st.sidebar:
    st.header("Setup")

    # Display API key status
    if google_api_key:
        st.success("Google API Key found in .env file")
    else:
        st.error(
            "Google API Key not found. Create a .env file with GOOGLE_API_KEY=your_api_key"
        )

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
        This app allows you to chat with JavaScript-dependent website content using:
        - Streamlit for the UI
        - Selenium for loading JavaScript-dependent websites
        - Langchain for document processing
        - Google's Gemini for understanding and responding
        - FAISS for efficient vector storage
    """
    )

# Display chat interface
st.subheader("Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
user_query = st.chat_input("Ask something about the website...")

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
            if not st.session_state.url:
                st.warning("Please enter a URL and process it first.")
            else:
                st.warning("Processing website. Please wait...")
elif not google_api_key:
    st.warning("Google API Key not found in .env file")

# Add a button to clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    if st.session_state.conversation:
        st.session_state.conversation.memory.clear()
