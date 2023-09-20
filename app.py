import streamlit as st
import Utils
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


#sidebar contents
with st.sidebar:
    st.title("Document Reader")
    st.markdown("""
    This app is an LLM-Powered built using:
    - streamlt
    - LangChain
    - OpenAI
    - HuggingFace 
    - PineCone
    """)
    add_vertical_space(6)
    st.write("Document Reader makes your work easier")

def split_doc(text, chunk_size=1000,chunk_overlap=100 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text)
    return docs

def vector_Embedding(docs, embedding_model):
    pinecone.init(api_key = os.getenv("api_key"), environment= os.getenv("environment") )
    index = Pinecone.from_texts(docs, embedding_model, index_name=os.getenv("index_name"))
    return index

def get_similar_docs(query, index):
    similar_docs = index.similarity_search(query, k=2)
    return similar_docs

def get_answer(query, similar_docs):
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    answer =  chain.run(input_documents=similar_docs, question=query)
    return  answer

def main():
    st.title("Chat with PDF") 
    load_dotenv(".env")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = "pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)  
        docs = split_doc(text, chunk_size=1000,chunk_overlap=100)
        # st.write(docs) 

        # Embedding
        embeddings_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
        index = vector_Embedding(docs, embeddings_model)

        # get similardoc
        query = st.text_area("Enter your question")
        button = st.button("Submit")
        
        if query and button:
            with st.spinner("Reading the Document..."):
                similar_docs = get_similar_docs(query, index)
                st.expander("Context").write(similar_docs)

                add_vertical_space(3)
                # Q&A using OpenAI
                answer = get_answer(query, similar_docs)
                st.write("Answer : "+ answer)

   

if __name__=="__main__":
    main()