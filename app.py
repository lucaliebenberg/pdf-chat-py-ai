# from langchain.vectorstores.cassandra import Cassandra
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
# from langchain_openai import OpenAI, OpenAIEmbeddings

# from dotenv import load_dotenv
# load_dotenv()


# ## Support for dataset retrieval with Hugging Face
# from datasets import load_dataset

# ## With CassIO, the engine powering the Astra DB integration in Langchain, 
# ## you will also initialize the DB connection
# import cassio

# from PyPDF2 import PdfReader
# import os


# # Env variables
# ASTRA_DB_APPLICATION_TOKEN=os.environ['ASTRA_DB_APPLICATION_TOKEN']
# ASTRA_DB_ID=os.environ['ASTRA_DB_ID']
# OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

# # path to pdf
# pdf_reader=PdfReader('xbox.pdf')

# from typing_extensions import Concatenate

# # read etxt from pdf
# raw_text=''
# for i, page in enumerate(pdf_reader.pages):
#     content=page.extract_text()
#     if content:
#         raw_text+=content


# ## initialize DB connection
# cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# ## create Langchain embedding & LLM objects for later usage
# llm=OpenAI(openai_api_key=OPENAI_API_KEY)
# embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # create Langchain vector store
# astra_vector_store=Cassandra(
#     embedding=embedding,
#     table_name="pdf_qa_demo",
#     session=None,
#     keyspace=None,
# )

# # split the text using CharacterTextSplit
# from langchain.text_splitter import CharacterTextSplitter
# text_splitter=CharacterTextSplitter(
#     separator="\n",
#     chunk_size=800,
#     chunk_overlap=200,
#     length_function=len,
# )
# texts=text_splitter.split_text(raw_text)

# ## insert text into DB
# astra_vector_store.add_texts(texts[:50])

# print("Inserted %i headlines." % len(texts[:50]))

# astra_vector_index=VectorStoreIndexWrapper(vectorstore=astra_vector_store)


# # define questions for queries
# first_question=True

# while True:
#     if first_question:
#         query_text=input("\nEnter your question (or type 'quit' to exit): ").strip()
#     else:
#         query_text=input("\nWhat's your next question (or type 'quit' to exit): ").strip()
    
#     if query_text.lower() == "quit":
#         break

#     if query_text == "":
#         continue

#     first_question=False

#     print("\nQUESTION: \"%s\"" % query_text)
#     answer=astra_vector_index.query(query_text, llm=llm).strip()
#     print("\nANSWER: \"%s\"" % answer)

#     print("FIRST DOCUMENTS BY RELEVANCE: ")
#     for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
#         print("  [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))

import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_openai import OpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import cassio
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

question_index = 200

# Streamlit interface
def main():
    global question_index  # Use the global variable

    # Your existing Streamlit code here...

    # Ask questions
    first_question = True
    while True:
        if first_question:
            question_prompt = "Enter your question (or type 'quit' to exit):"
        else:
            question_prompt = "What's your next question (or type 'quit' to exit):"
        
        # Generate a unique key for the text input widget
        question_key = f"question_input_{question_index}"
        
        # Use the unique key for the text input widget
        query_text = st.text_input(question_prompt, key=question_key).strip()
        
        if query_text.lower() == "quit":
            break
        
        if query_text == "":
            continue
        
        first_question = False
        question_index += 1
        
            
        st.write("QUESTION:", query_text)
        answer = astra_vector_index.query(query_text, llm=llm).strip()
        st.write("ANSWER:", answer)
            
        st.write("FIRST DOCUMENTS BY RELEVANCE:")
        for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
            st.write("[%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))

if __name__ == "__main__":
    main()

