import pinecone
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings

def pdfloader(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text    

def split_doc(text, chunk_size=750,chunk_overlap=75 ):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(text)
    return docs

def vector_Embedding(docs):
    embedding_model = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    index = Chroma.from_texts(docs, embedding_model)
    return index

def get_similar_docs(query, index):
    similar_docs = index.similarity_search(query, k=2)
    return similar_docs

def get_answer(query, similar_docs):
    llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    answer =  chain.run(input_documents=similar_docs, question=query)
    return  answer
