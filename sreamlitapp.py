import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    raw_text = ""
    pdf_reader = PdfReader(file)
    for i, page in enumerate(pdf_reader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to initialize database connection and vector store
def initialize_database_and_vectorstore():
    # Env variables
    ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
    ASTRA_DB_ID = os.getenv('ASTRA_DB_ID')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # initialize DB connection
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    # create Langchain embedding & LLM objects for later usage
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # create Langchain vector store
    astra_vector_store = Cassandra(
        embedding=embedding,
        table_name="pdf_qa_demo",
        session=None,
        keyspace=None,
    )
    return astra_vector_store, llm

# Streamlit interface
def main():
    st.title("PDF Chat Application")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        raw_text = extract_text_from_pdf(uploaded_file)
        st.write("PDF text successfully extracted!")
        
        # Initialize database connection and vector store
        astra_vector_store, llm = initialize_database_and_vectorstore()
        
        # split the text using CharacterTextSplit
        from langchain.text_splitter import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # insert text into DB
        astra_vector_store.add_texts(texts[:50])

        st.write("Inserted %i headlines." % len(texts[:50]))

        astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

        # Ask questions
        first_question = True
        while True:
            if first_question:
                query_text = st.text_input("Enter your question (or type 'quit' to exit):").strip()
            else:
                query_text = st.text_input("What's your next question (or type 'quit' to exit):").strip()
            
            if query_text.lower() == "quit":
                break
            
            if query_text == "":
                continue
            
            first_question = False
            
            st.write("QUESTION:", query_text)
            answer = astra_vector_index.query(query_text, llm=llm).strip()
            st.write("ANSWER:", answer)
            
            st.write("FIRST DOCUMENTS BY RELEVANCE:")
            for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
                st.write("[%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))

if __name__ == "__main__":
    main()
