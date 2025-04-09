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

# Function to generate flashcards from document content
def generate_flashcards(llm, document_content, num_cards=10):
    """Generate flashcards from document content using LLM"""
    
    prompt = f"""
    Based on the following content, create {num_cards} flashcards in the format:
    Question: [Question text]
    Answer: [Answer text]
    
    Focus on key concepts, definitions, and important facts.
    Content: {document_content[:10000]}  # Limit content size for prompt
    """
    
    response = llm.invoke(prompt).content
    
    # Parse response into flashcard format
    flashcards = []
    lines = response.split('\n')
    current_question = None
    current_answer = ""
    
    for line in lines:
        if line.startswith("Question:"):
            if current_question:
                flashcards.append({"question": current_question, "answer": current_answer.strip()})
            current_question = line[len("Question:"):].strip()
            current_answer = ""
        elif line.startswith("Answer:"):
            current_answer = line[len("Answer:"):].strip()
        elif current_question and current_answer:
            current_answer += " " + line.strip()
    
    # Add the last flashcard
    if current_question:
        flashcards.append({"question": current_question, "answer": current_answer.strip()})
    
    return flashcards

# Function to generate quiz from document content
def generate_quiz(llm, document_content, num_questions=5, quiz_type="multiple_choice"):
    """Generate a quiz from document content using LLM"""
    
    prompt = f"""
    Based on the following content, create a {quiz_type} quiz with {num_questions} questions.
    
    For multiple choice questions, use this format:
    Question: [Question text]
    A. [Option A]
    B. [Option B]
    C. [Option C]
    D. [Option D]
    Correct Answer: [Letter of correct answer]
    Explanation: [Brief explanation of why this is the correct answer]
    
    For short answer questions, use this format:
    Question: [Question text]
    Answer: [Correct answer]
    
    Focus on testing understanding of key concepts and facts.
    Content: {document_content[:10000]}  # Limit content size for prompt
    """
    
    response = llm.invoke(prompt).content
    
    # Parse response into quiz format
    quiz_questions = []
    current_question = {}
    parsing_state = None
    
    for line in response.split('\n'):
        if line.startswith("Question:"):
            if current_question and "question" in current_question:
                quiz_questions.append(current_question)
                current_question = {}
            
            current_question["question"] = line[len("Question:"):].strip()
            current_question["options"] = []
            parsing_state = "question"
            
        elif line.startswith(("A.", "B.", "C.", "D.")) and parsing_state == "question":
            option_letter = line[0]
            option_text = line[2:].strip()
            current_question["options"].append({"letter": option_letter, "text": option_text})
            
        elif line.startswith("Correct Answer:"):
            current_question["correct_answer"] = line[len("Correct Answer:"):].strip()
            parsing_state = "correct_answer"
            
        elif line.startswith("Explanation:"):
            current_question["explanation"] = line[len("Explanation:"):].strip()
            parsing_state = "explanation"
            
        elif line.startswith("Answer:") and "options" not in current_question:
            current_question["answer"] = line[len("Answer:"):].strip()
            parsing_state = "answer"
            
        elif parsing_state == "explanation" and line.strip():
            current_question["explanation"] += " " + line.strip()
            
        elif parsing_state == "answer" and line.strip():
            current_question["answer"] += " " + line.strip()
    
    # Add the last question
    if current_question and "question" in current_question:
        quiz_questions.append(current_question)
    
    return quiz_questions

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
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []
if "flashcards" not in st.session_state:
    st.session_state.flashcards = []
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "current_card_index" not in st.session_state:
    st.session_state.current_card_index = 0
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

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
                            
                            # Store chunks for study tools
                            st.session_state.document_chunks = chunks

                            # Create embeddings and index
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", google_api_key=google_api_key
                            )
                            vectorstore = FAISS.from_documents(chunks, embeddings)

                            # Create conversation chain
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )

                            # Initialize LLM
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.7,
                                google_api_key=google_api_key,
                            )

                            st.session_state.conversation = (
                                ConversationalRetrievalChain.from_llm(
                                    llm=llm,
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
                            
                            # Reset study tools
                            st.session_state.flashcards = []
                            st.session_state.quiz = []
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
                            
                            # Store chunks for study tools
                            st.session_state.document_chunks = chunks
                            
                            # Create embeddings and index
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", google_api_key=google_api_key
                            )
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                            
                            # Create conversation chain
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )
                            
                            # Initialize LLM
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.7,
                                google_api_key=google_api_key,
                            )
                            
                            st.session_state.conversation = (
                                ConversationalRetrievalChain.from_llm(
                                    llm=llm,
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
                            
                            # Reset study tools
                            st.session_state.flashcards = []
                            st.session_state.quiz = []
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            else:
                if not google_api_key:
                    st.warning("Google API Key not found in .env file")
                if not uploaded_files:
                    st.warning("Please upload at least one document")

    # Study Tools section in sidebar
    if st.session_state.document_chunks:
        st.markdown("---")
        st.header("Study Tools")
        
        with st.expander("Flashcard Generator"):
            num_cards = st.slider("Number of Flashcards", 5, 20, 10)
            if st.button("Generate Flashcards"):
                if st.session_state.conversation:
                    with st.spinner("Generating flashcards..."):
                        try:
                            # Prepare document content for generation
                            document_content = " ".join([
                                chunk.page_content for chunk in st.session_state.document_chunks[:20]
                            ])
                            
                            # Get LLM from conversation chain
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.7,
                                google_api_key=google_api_key,
                            )
                            
                            # Generate flashcards
                            flashcards = generate_flashcards(llm, document_content, num_cards)
                            st.session_state.flashcards = flashcards
                            st.session_state.current_card_index = 0
                            st.success(f"Generated {len(flashcards)} flashcards!")
                        except Exception as e:
                            st.error(f"Error generating flashcards: {str(e)}")
        
        with st.expander("Quiz Generator"):
            num_questions = st.slider("Number of Questions", 3, 15, 5)
            quiz_type = st.selectbox("Quiz Type", ["multiple_choice", "short_answer"])
            
            if st.button("Generate Quiz"):
                if st.session_state.conversation:
                    with st.spinner("Generating quiz..."):
                        try:
                            # Prepare document content for generation
                            document_content = " ".join([
                                chunk.page_content for chunk in st.session_state.document_chunks[:20]
                            ])
                            
                            # Get LLM from conversation chain
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.7,
                                google_api_key=google_api_key,
                            )
                            
                            # Generate quiz
                            quiz = generate_quiz(llm, document_content, num_questions, quiz_type)
                            st.session_state.quiz = quiz
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.success(f"Generated quiz with {len(quiz)} questions!")
                        except Exception as e:
                            st.error(f"Error generating quiz: {str(e)}")

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
        - Study tools: flashcards and quizzes for better learning
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

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Chat", "Flashcards", "Quiz"])

# Chat Tab
with tab1:
    st.subheader("Chat with Content")
    
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
    
    # Add a button to clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        if st.session_state.conversation:
            st.session_state.conversation.memory.clear()

# Flashcards Tab
with tab2:
    st.subheader("Flashcards")
    
    if st.session_state.flashcards:
        current_card = st.session_state.current_card_index
        
        # Card navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous"):
                st.session_state.current_card_index = max(0, current_card - 1)
                st.rerun()
        with col3:
            if st.button("Next ➡️"):
                st.session_state.current_card_index = min(len(st.session_state.flashcards) - 1, current_card + 1)
                st.rerun()
        
        # Display card
        card = st.session_state.flashcards[current_card]
        st.info(f"Card {current_card + 1} of {len(st.session_state.flashcards)}")
        
        # Flashcard display with flip functionality
        with st.container():
            st.markdown(f"### {card['question']}")
            show_answer = st.checkbox("Show Answer", key=f"show_answer_{current_card}")
            if show_answer:
                st.success(card['answer'])
    else:
        if st.session_state.document_chunks:
            st.info("Generate flashcards from the sidebar to start studying!")
        else:
            st.warning("Process a website or documents first to create flashcards.")

# Quiz Tab
with tab3:
    st.subheader("Quiz")
    
    if st.session_state.quiz:
        # Display the quiz
        if not st.session_state.quiz_submitted:
            for i, question in enumerate(st.session_state.quiz):
                st.markdown(f"### Question {i+1}")
                st.markdown(question["question"])
                
                if "options" in question and question["options"]:  # Multiple choice
                    options = {opt["letter"]: opt["text"] for opt in question["options"]}
                    selected = st.radio(
                        f"Select answer for question {i+1}:", 
                        options.keys(),
                        format_func=lambda x: f"{x}. {options[x]}",
                        key=f"q_{i}"
                    )
                    st.session_state.quiz_answers[i] = selected
                else:  # Short answer
                    answer = st.text_input(
                        f"Your answer for question {i+1}:", 
                        key=f"q_{i}"
                    )
                    st.session_state.quiz_answers[i] = answer
                
                st.markdown("---")
            
            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            # Show results
            correct_count = 0
            
            for i, question in enumerate(st.session_state.quiz):
                st.markdown(f"### Question {i+1}")
                st.markdown(question["question"])
                
                user_answer = st.session_state.quiz_answers.get(i, "")
                
                if "options" in question and question["options"]:  # Multiple choice
                    options = {opt["letter"]: opt["text"] for opt in question["options"]}
                    correct = user_answer == question.get("correct_answer", "")
                    
                    for letter, text in options.items():
                        if letter == user_answer:
                            if correct:
                                st.success(f"{letter}. {text} ✓ (Your answer)")
                            else:
                                st.error(f"{letter}. {text} ✗ (Your answer)")
                        elif letter == question.get("correct_answer", ""):
                            st.success(f"{letter}. {text} ✓ (Correct answer)")
                        else:
                            st.write(f"{letter}. {text}")
                    
                    if "explanation" in question:
                        st.info(f"Explanation: {question['explanation']}")
                    
                    if correct:
                        correct_count += 1
                else:  # Short answer
                    st.write(f"Your answer: {user_answer}")
                    st.write(f"Correct answer: {question.get('answer', '')}")
                    
                    # Simple string matching - could use more sophisticated comparison
                    correct = user_answer.lower() == question.get('answer', '').lower()
                    if correct:
                        st.success("Correct! ✓")
                        correct_count += 1
                    else:
                        st.error("Incorrect ✗")
                
                st.markdown("---")
            
            score_percentage = int((correct_count / len(st.session_state.quiz)) * 100) if st.session_state.quiz else 0
            st.markdown(f"## Your Score: {correct_count}/{len(st.session_state.quiz)} ({score_percentage}%)")
            
            if st.button("Try Again"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}
                st.rerun()
    else:
        if st.session_state.document_chunks:
            st.info("Generate a quiz from the sidebar to start testing your knowledge!")
        else:
            st.warning("Process a website or documents first to create a quiz.")