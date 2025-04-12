# Contextify : Chat with Websites, Documents, and SQL Databases using Natural Language


## 📖 Overview

Contextify is a powerful RAG (Retrieval-Augmented Generation) application that lets you interact with various types of content using natural language. Whether you need to extract information from websites, learn from documents, or query SQL databases without writing SQL code, Contextify makes it simple with its intuitive chat interface.

### Key Features

- **Website Chat**: Extract and query content from any website, including JavaScript-rendered pages
- **Document Chat**: Upload and ask questions about various document formats including PDFs, Word docs, and more  
- **Study Tools**: Generate flashcards and quizzes automatically from your documents
- **SQL Assistant**: Connect to MySQL databases and query them using natural language
- **RAG Architecture**: Uses Google's Gemini models for high-quality, context-aware responses

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Google Gemini API key
- MySQL (for database features)
- Chrome/Chromium (for website scraping)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/contextify.git
   cd contextify
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root directory and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

Start the Streamlit app:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

## 📊 Usage

### Website Interaction

1. Select "Website" as the source type in the sidebar
2. Enter the URL of the website you want to chat about
3. Adjust the JavaScript load wait time if needed
4. Click "Process Website" and wait for the content to be processed
5. Ask questions about the website content in the chat interface

### Document Analysis

1. Select "Document" as the source type
2. Upload one or more supported documents (PDF, DOCX, TXT, CSV, PPTX, HTML)
3. Click "Process Document(s)" and wait for the content to be processed
4. Chat with your documents or use the study tools:
   - **Flashcards**: Generate and flip through flashcards based on document content
   - **Quiz**: Test your knowledge with auto-generated multiple-choice or short-answer questions

### SQL Database Querying

1. Select "SQL Database" as the source type
2. Enter your MySQL database connection details
3. Connect to your database and select the specific database you want to work with
4. Chat with your database using natural language or use the SQL Explorer:
   - Ask questions like "Show me the top 5 customers by total purchase amount"
   - View the auto-generated SQL and execution results
   - Access your query history

## 🔍 Supported File Formats

- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Microsoft PowerPoint (`.pptx`, `.ppt`)
- Plain Text (`.txt`)
- CSV (`.csv`)
- HTML (`.html`, `.htm`)

## ⚙️ Advanced Configuration

Access advanced settings in the sidebar:
- Customize the embedding model
- Change the LLM model
- Adjust study tool parameters

## 🧠 How It Works

Contextify uses a sophisticated RAG architecture:

1. **Loading**: Content is extracted from websites (using Selenium for JavaScript rendering), documents (using specialized loaders), or databases (via connection APIs)
2. **Processing**: Text is split into chunks and embedded using Google's embedding models
3. **Indexing**: Embeddings are stored using FAISS for efficient vector search
4. **Retrieval**: When you ask a question, relevant chunks are retrieved based on semantic similarity
5. **Generation**: Google's Gemini models generate human-like responses based on the retrieved context

For SQL interactions, natural language is translated into valid SQL queries which are then executed against your database.

## 📝 Technical Details

- **Frontend**: Streamlit
- **Web Scraping**: Selenium with Chrome WebDriver
- **Document Processing**: LangChain document loaders
- **Vector Search**: FAISS
- **Language Model**: Google Generative AI (Gemini)
- **Database Connectivity**: MySQL Connector
