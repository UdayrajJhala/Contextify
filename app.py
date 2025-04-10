import streamlit as st
import os
import time
import tempfile
import mysql.connector
from mysql.connector import Error
import pandas as pd
from sqlalchemy import create_engine, inspect
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
st.set_page_config(page_title="Chat with Websites, Documents & SQL", page_icon="🌐", layout="wide")

# App title and description
st.title("🌐 Chat with Websites, Documents & SQL")
st.markdown(
    """
    Enter a URL to chat with a website, upload documents to ask questions about their content,
    or connect to a MySQL database to query data using natural language.
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
    # ... existing code unchanged ...
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
    # ... existing code unchanged ...
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
    # ... existing code unchanged ...
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
    # ... existing code unchanged ...
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

# New function to connect to MySQL database
def connect_to_mysql(host, user, password, database=None):
    """Connect to MySQL database and return connection"""
    try:
        if database:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database
            )
        else:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password
            )
        
        if connection.is_connected():
            return connection, None
        
    except Error as e:
        return None, str(e)
    
    return None, "Failed to connect to the database."

# New function to get database schema information
def get_database_schema(connection):
    """Extract database schema information from connection"""
    cursor = connection.cursor()
    schema_info = {}
    
    # Get list of databases
    cursor.execute("SHOW DATABASES")
    databases = [db[0] for db in cursor.fetchall()]
    
    # Get current database
    cursor.execute("SELECT DATABASE()")
    current_db = cursor.fetchone()[0]
    
    if current_db:
        # Get tables in the current database
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        # Get schema for each table
        table_schemas = {}
        for table in tables:
            cursor.execute(f"DESCRIBE `{table}`")
            columns = cursor.fetchall()
            
            # Format column information
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col[0],
                    "type": col[1],
                    "null": col[2],
                    "key": col[3],
                    "default": col[4],
                    "extra": col[5]
                })
            
            table_schemas[table] = column_info
            
            # Get sample data for better context
            try:
                cursor.execute(f"SELECT * FROM `{table}` LIMIT 3")
                sample_rows = cursor.fetchall()
                
                if sample_rows:
                    # Get column names
                    column_names = [i[0] for i in cursor.description]
                    # Format sample data
                    sample_data = []
                    for row in sample_rows:
                        sample_row = {}
                        for i, col_name in enumerate(column_names):
                            sample_row[col_name] = row[i]
                        sample_data.append(sample_row)
                    
                    table_schemas[table + "_sample"] = sample_data
            except:
                # Skip sample data if there's an error
                pass
        
        schema_info["current_database"] = current_db
        schema_info["tables"] = table_schemas
    
    schema_info["databases"] = databases
    
    cursor.close()
    return schema_info

# New function to execute SQL query
def execute_sql_query(connection, query):
    """Execute SQL query and return results"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        
        # Check if the query returns results
        if cursor.description:
            columns = [i[0] for i in cursor.description]
            results = cursor.fetchall()
            
            # Convert results to list of dictionaries
            formatted_results = []
            for row in results:
                formatted_row = {}
                for i, col_name in enumerate(columns):
                    formatted_row[col_name] = row[i]
                formatted_results.append(formatted_row)
            
            return {"success": True, "data": formatted_results, "columns": columns, "affected_rows": len(formatted_results)}
        else:
            # For INSERT, UPDATE, DELETE statements
            affected_rows = cursor.rowcount
            return {"success": True, "affected_rows": affected_rows}
            
    except Error as e:
        return {"success": False, "error": str(e)}
    finally:
        cursor.close()

# New function to convert natural language to SQL
def natural_language_to_sql(llm, schema_info, question):
    """Convert natural language question to SQL query using LLM"""
    # Create a detailed prompt about the database schema
    schema_prompt = "Database schema:\n"
    
    if "current_database" in schema_info:
        schema_prompt += f"Current database: {schema_info['current_database']}\n\n"
        
        if "tables" in schema_info:
            for table_name, columns in schema_info["tables"].items():
                if not table_name.endswith("_sample"):
                    schema_prompt += f"Table: {table_name}\n"
                    schema_prompt += "Columns:\n"
                    
                    for col in columns:
                        schema_prompt += f"- {col['name']} ({col['type']})"
                        if col['key'] == 'PRI':
                            schema_prompt += " PRIMARY KEY"
                        elif col['key'] == 'MUL':
                            schema_prompt += " INDEX"
                        schema_prompt += "\n"
                    
                    # Add sample data if available
                    sample_key = f"{table_name}_sample"
                    if sample_key in schema_info["tables"] and schema_info["tables"][sample_key]:
                        schema_prompt += "Sample data:\n"
                        for i, row in enumerate(schema_info["tables"][sample_key][:3]):
                            schema_prompt += f"Row {i+1}: {row}\n"
                    
                    schema_prompt += "\n"
    
    # Create the full prompt for the LLM
    prompt = f"""
You are a SQL expert. Given the following database schema, convert the user's natural language question into a valid MySQL SQL query.
Only return the SQL query without any explanations.

{schema_prompt}

User question: {question}

SQL query:
"""
    
    try:
        response = llm.invoke(prompt).content
        # Clean up the response to extract just the SQL query
        sql_query = response.strip()
        
        # Remove markdown code block formatting if present
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        return {"success": True, "query": sql_query}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Function to explain the SQL query
def explain_sql_query(llm, query, schema_info, question):
    """Explain the generated SQL query in natural language"""
    schema_prompt = "Database schema summary:\n"
    
    if "current_database" in schema_info:
        schema_prompt += f"Database: {schema_info['current_database']}\n"
        schema_prompt += f"Tables: {', '.join([t for t in schema_info.get('tables', {}).keys() if not t.endswith('_sample')])}\n\n"
    
    prompt = f"""
Given this SQL query and database information, explain in simple terms what the query does and how it answers the user's question.
Be concise but thorough, explaining any joins, conditions, or aggregations.

{schema_prompt}

User question: {question}

SQL query: {query}

Explanation:
"""
    
    try:
        response = llm.invoke(prompt).content
        return response.strip()
    except Exception as e:
        return f"Error explaining query: {str(e)}"

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
if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = None  
# New session state variables for SQL functionality
if "db_connection" not in st.session_state:
    st.session_state.db_connection = None
if "db_schema" not in st.session_state:
    st.session_state.db_schema = None
if "db_name" not in st.session_state:
    st.session_state.db_name = ""
if "sql_history" not in st.session_state:
    st.session_state.sql_history = []

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
    source_type = st.radio("Select Source Type:", ["Website", "Document", "SQL Database"])
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
                st.session_state.db_name = ""  # Clear database name

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

    elif source_type == "Document":  # Document upload section
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
                            st.session_state.db_name = ""  # Clear database name

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

    else:  # SQL Database section
        st.write("Connect to MySQL Database")

        # Database connection form
        with st.form("db_connection_form"):
            db_host = st.text_input("Host", "localhost")
            db_user = st.text_input("Username", "root")
            db_password = st.text_input("Password", type="password")
            db_name = st.text_input("Database Name (optional)", "")

            submit_button = st.form_submit_button("Connect to Database")

            if submit_button:
                if google_api_key:
                    with st.spinner("Connecting to database..."):
                        connection, error = connect_to_mysql(db_host, db_user, db_password, db_name if db_name else None)

                        if connection and connection.is_connected():
                            st.session_state.db_connection = connection
                            st.session_state.db_name = db_name if db_name else "MySQL"
                            st.session_state.url = ""  # Clear URL
                            st.session_state.document_name = ""  # Clear document name

                            # Get database schema information
                            schema_info = get_database_schema(connection)
                            st.session_state.db_schema = schema_info

                            # Initialize LLM
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.2,  # Lower temperature for more deterministic SQL generation
                                google_api_key=google_api_key,
                            )

                            # Create memory for conversation
                            memory = ConversationBufferMemory(
                                memory_key="chat_history", return_messages=True
                            )

                            # Create conversation chain (without a retriever)
                            st.session_state.conversation = None  # Clear any existing conversation

                            st.success(f"Connected to database successfully! You can now chat about your database.")

                            # Clear previous chat
                            st.session_state.chat_history = []
                            st.session_state.messages = []
                            st.session_state.sql_history = []
                        else:
                            st.error(f"Database connection failed: {error}")
                else:
                    st.warning("Google API Key not found in .env file")

        # Show database selection if connected
        if st.session_state.db_connection and st.session_state.db_schema:
            st.write("Database Selection")

            # Get available databases
            available_dbs = st.session_state.db_schema.get("databases", [])
            current_db = st.session_state.db_schema.get("current_database", "")

            # Create dropdown for database selection
            selected_db = st.selectbox(
                "Select Database", 
                available_dbs,
                index=available_dbs.index(current_db) if current_db in available_dbs else 0
            )

            if st.button("Switch Database"):
                if selected_db and selected_db != current_db:
                    try:
                        # Execute USE statement
                        cursor = st.session_state.db_connection.cursor()
                        cursor.execute(f"USE `{selected_db}`")
                        cursor.close()

                        # Update schema information
                        schema_info = get_database_schema(st.session_state.db_connection)
                        st.session_state.db_schema = schema_info
                        st.session_state.db_name = selected_db

                        st.success(f"Switched to database: {selected_db}")
                    except Error as e:
                        st.error(f"Error switching database: {str(e)}")

    # Study Tools section in sidebar (for Website and Document sources)
    if source_type in ["Website", "Document"] and st.session_state.document_chunks:
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
        This app allows you to chat with website content, documents, and SQL databases using:
        - Streamlit for the UI
        - Selenium for loading JavaScript-dependent websites
        - Document loaders for various file formats (PDF, DOCX, TXT, CSV, PPTX, HTML)
        - Langchain for document processing
        - Google's Gemini for understanding and responding
        - FAISS for efficient vector storage
        - Study tools: flashcards and quizzes for better learning
        - MySQL database querying with natural language
        """
    )

# Display current source information
if st.session_state.url:
    st.info(f"Currently loaded source: Website - {st.session_state.url}")
elif st.session_state.document_name:
    st.info(f"Currently loaded source: Document(s) - {st.session_state.document_name}")
elif st.session_state.db_name:
    st.info(f"Currently loaded source: Database - {st.session_state.db_name}")
else:
    source_type = st.session_state.source_type or "Website"
    if source_type == "Website":
        st.warning("No website processed yet. Please enter a URL and process it.")
    elif source_type == "Document":
        st.warning("No documents processed yet. Please upload documents and process them.")
    else:
        st.warning("No database connected yet. Please connect to a database.")

# Create tabs based on source type
if st.session_state.source_type == "Website":
    # For website source, only show Chat tab
    tab1 = st.tabs(["Chat"])[0]

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

            # Process with normal RAG conversation for website
            if st.session_state.conversation:
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
                        st.warning("Processing content. Please wait...")

        # Add a button to clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            if st.session_state.conversation and hasattr(
                st.session_state.conversation, "memory"
            ):
                st.session_state.conversation.memory.clear()

elif st.session_state.source_type == "Document":
    # For document source, show Chat, Flashcards, and Quiz tabs
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

            # Process with normal RAG conversation for documents
            if st.session_state.conversation:
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
                    if not st.session_state.document_name:
                        st.warning("Please upload documents and process them first.")
                    else:
                        st.warning("Processing content. Please wait...")

        # Add a button to clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            if st.session_state.conversation and hasattr(
                st.session_state.conversation, "memory"
            ):
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
                    st.session_state.current_card_index = min(
                        len(st.session_state.flashcards) - 1, current_card + 1
                    )
                    st.rerun()

            # Display card
            card = st.session_state.flashcards[current_card]
            st.info(f"Card {current_card + 1} of {len(st.session_state.flashcards)}")

            # Flashcard display with flip functionality
            with st.container():
                st.markdown(f"### {card['question']}")
                show_answer = st.checkbox(
                    "Show Answer", key=f"show_answer_{current_card}"
                )
                if show_answer:
                    st.success(card["answer"])
        else:
            if st.session_state.document_chunks:
                st.info("Generate flashcards from the sidebar to start studying!")
            else:
                st.warning("Process documents first to create flashcards.")

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
                        options = {
                            opt["letter"]: opt["text"] for opt in question["options"]
                        }
                        selected = st.radio(
                            f"Select answer for question {i+1}:",
                            options.keys(),
                            format_func=lambda x: f"{x}. {options[x]}",
                            key=f"q_{i}",
                        )
                        st.session_state.quiz_answers[i] = selected
                    else:  # Short answer
                        answer = st.text_input(
                            f"Your answer for question {i+1}:", key=f"q_{i}"
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
                        options = {
                            opt["letter"]: opt["text"] for opt in question["options"]
                        }
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
                        correct = (
                            user_answer.lower() == question.get("answer", "").lower()
                        )
                        if correct:
                            st.success("Correct! ✓")
                            correct_count += 1
                        else:
                            st.error("Incorrect ✗")

                    st.markdown("---")

                score_percentage = (
                    int((correct_count / len(st.session_state.quiz)) * 100)
                    if st.session_state.quiz
                    else 0
                )
                st.markdown(
                    f"## Your Score: {correct_count}/{len(st.session_state.quiz)} ({score_percentage}%)"
                )

                if st.button("Try Again"):
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_answers = {}
                    st.rerun()
        else:
            if st.session_state.document_chunks:
                st.info(
                    "Generate a quiz from the sidebar to start testing your knowledge!"
                )
            else:
                st.warning("Process documents first to create a quiz.")

else:  # SQL Database
    # For SQL source, show Chat and SQL Explorer tabs
    tab1, tab4 = st.tabs(["Chat", "SQL Explorer"])

    # Chat Tab
    with tab1:
        st.subheader("Chat with SQL Database")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Get user input
        user_query = st.chat_input("Ask questions about your database...")

        # Process user input
        if user_query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

            # Display user message
            with st.chat_message("user"):
                st.write(user_query)

            # Process as SQL query using natural language
            if st.session_state.db_connection:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            # Initialize LLM
                            llm = ChatGoogleGenerativeAI(
                                model="gemini-1.5-flash",
                                temperature=0.2,
                                google_api_key=google_api_key,
                            )

                            # Convert natural language to SQL
                            nl_to_sql_result = natural_language_to_sql(
                                llm, st.session_state.db_schema, user_query
                            )

                            if nl_to_sql_result["success"]:
                                sql_query = nl_to_sql_result["query"]

                                # Show the generated SQL
                                st.markdown("**Generated SQL Query:**")
                                st.code(sql_query, language="sql")

                                # Execute the query
                                query_result = execute_sql_query(
                                    st.session_state.db_connection, sql_query
                                )

                                if query_result["success"]:
                                    # Explain the query
                                    explanation = explain_sql_query(
                                        llm,
                                        sql_query,
                                        st.session_state.db_schema,
                                        user_query,
                                    )
                                    st.markdown("**Explanation:**")
                                    st.write(explanation)

                                    # Display results
                                    if "data" in query_result and query_result["data"]:
                                        st.markdown("**Query Results:**")
                                        # Convert to pandas DataFrame for display
                                        result_df = pd.DataFrame(query_result["data"])
                                        st.dataframe(result_df)

                                        # Save query and results in SQL history
                                        st.session_state.sql_history.append(
                                            {
                                                "question": user_query,
                                                "query": sql_query,
                                                "results": query_result["data"],
                                                "explanation": explanation,
                                            }
                                        )

                                        # Formulate response message
                                        result_message = f"{explanation}\n\nThe query returned {query_result['affected_rows']} rows."
                                        st.session_state.messages.append(
                                            {
                                                "role": "assistant",
                                                "content": result_message,
                                            }
                                        )

                                    else:
                                        # For non-SELECT queries
                                        result_message = f"{explanation}\n\nQuery executed successfully. {query_result['affected_rows']} rows affected."
                                        st.write(result_message)
                                        st.session_state.messages.append(
                                            {
                                                "role": "assistant",
                                                "content": result_message,
                                            }
                                        )

                                        # Save in SQL history
                                        st.session_state.sql_history.append(
                                            {
                                                "question": user_query,
                                                "query": sql_query,
                                                "affected_rows": query_result[
                                                    "affected_rows"
                                                ],
                                                "explanation": explanation,
                                            }
                                        )
                                else:
                                    # Query execution failed
                                    error_message = f"I generated this SQL query:\n```sql\n{sql_query}\n```\n\nBut there was an error executing it: {query_result['error']}"
                                    st.error(error_message)
                                    st.session_state.messages.append(
                                        {"role": "assistant", "content": error_message}
                                    )
                            else:
                                # Failed to generate SQL
                                error_message = f"I couldn't generate SQL for your question: {nl_to_sql_result['error']}"
                                st.error(error_message)
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": error_message}
                                )

                        except Exception as e:
                            error_message = f"Error generating response: {str(e)}"
                            st.error(error_message)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_message}
                            )
            else:
                with st.chat_message("assistant"):
                    st.warning("Please connect to a database first via the sidebar.")

        # Add a button to clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # SQL Explorer Tab
    with tab4:
        st.subheader("SQL Explorer")

        if st.session_state.db_connection and st.session_state.db_schema:
            # Display database schema
            with st.expander("Database Schema", expanded=False):
                if "current_database" in st.session_state.db_schema:
                    current_db = st.session_state.db_schema["current_database"]
                    st.markdown(f"### Current Database: {current_db}")

                    if "tables" in st.session_state.db_schema:
                        tables = [
                            t
                            for t in st.session_state.db_schema["tables"].keys()
                            if not t.endswith("_sample")
                        ]

                        for table in tables:
                            st.markdown(f"#### Table: {table}")

                            # Convert column info to DataFrame for display
                            columns = st.session_state.db_schema["tables"][table]
                            column_data = []
                            for col in columns:
                                column_data.append(
                                    {
                                        "Name": col["name"],
                                        "Type": col["type"],
                                        "NULL": "YES" if col["null"] == "YES" else "NO",
                                        "Key": col["key"] if col["key"] else "",
                                        "Default": (
                                            col["default"] if col["default"] else ""
                                        ),
                                        "Extra": col["extra"] if col["extra"] else "",
                                    }
                                )

                            if column_data:
                                df = pd.DataFrame(column_data)
                                st.dataframe(df)

            # Natural language to SQL
            st.markdown("### Natural Language to SQL")
            nl_question = st.text_input(
                "Ask a question about your data",
                placeholder="Show me the top 5 customers by order value",
            )

            if st.button("Convert to SQL") and nl_question:
                with st.spinner("Generating SQL..."):
                    try:
                        # Initialize LLM
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-1.5-flash",
                            temperature=0.2,
                            google_api_key=google_api_key,
                        )

                        # Convert natural language to SQL
                        nl_to_sql_result = natural_language_to_sql(
                            llm, st.session_state.db_schema, nl_question
                        )

                        if nl_to_sql_result["success"]:
                            st.session_state.generated_sql = nl_to_sql_result["query"]
                        else:
                            st.error(
                                f"Error generating SQL: {nl_to_sql_result['error']}"
                            )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # Display the generated SQL (only once)
            if st.session_state.generated_sql:
                st.code(st.session_state.generated_sql, language="sql")

            # Second button to execute the generated SQL
            if st.button("Execute Generated SQL") and st.session_state.generated_sql:
                with st.spinner("Executing query..."):
                    query_result = execute_sql_query(
                        st.session_state.db_connection, st.session_state.generated_sql
                    )

                    if query_result["success"]:
                        if "data" in query_result and query_result["data"]:
                            st.success(
                                f"Query executed successfully. {query_result['affected_rows']} rows returned."
                            )

                            # Convert to pandas DataFrame for display
                            result_df = pd.DataFrame(query_result["data"])
                            st.dataframe(result_df)

                            # Save to history
                            st.session_state.sql_history.append(
                                {
                                    "question": nl_question,
                                    "query": st.session_state.generated_sql,
                                    "results": query_result["data"],
                                }
                            )
                        else:
                            st.success(
                                f"Query executed successfully. {query_result['affected_rows']} rows affected."
                            )

                            # Save to history
                            st.session_state.sql_history.append(
                                {
                                    "question": nl_question,
                                    "query": st.session_state.generated_sql,
                                    "affected_rows": query_result["affected_rows"],
                                }
                            )
                    else:
                        st.error(f"Error executing query: {query_result['error']}")

            # SQL History
            if st.session_state.sql_history:
                with st.expander("SQL Query History", expanded=False):
                    for i, entry in enumerate(reversed(st.session_state.sql_history)):
                        st.markdown(
                            f"### Query {len(st.session_state.sql_history) - i}"
                        )

                        if "question" in entry:
                            st.markdown(f"**Question:** {entry['question']}")

                        st.markdown("**SQL:**")
                        st.code(entry["query"], language="sql")

                        if "explanation" in entry:
                            st.markdown(f"**Explanation:** {entry['explanation']}")

                        if "results" in entry and entry["results"]:
                            st.markdown("**Results:**")
                            result_df = pd.DataFrame(entry["results"])
                            st.dataframe(result_df)
                        elif "affected_rows" in entry:
                            st.markdown(f"**Affected rows:** {entry['affected_rows']}")

                        st.markdown("---")
        else:
            st.warning("Please connect to a database first via the sidebar.")

# Disconnect database when app stops
def disconnect_db():
    if st.session_state.db_connection:
        try:
            st.session_state.db_connection.close()
            print("Database connection closed")
        except:
            pass

# Register the disconnect function to run when the app stops
import atexit
atexit.register(disconnect_db)
