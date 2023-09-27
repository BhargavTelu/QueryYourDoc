import streamlit as st
from Utils import *
import Utils
from dotenv import load_dotenv
load_dotenv(".env")
from streamlit_extras.add_vertical_space import add_vertical_space

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

def main():
    st.title("Chat with PDF") 
    load_dotenv(".env")
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type = "pdf")
    if pdf is not None:
        text = Utils.pdfloader(pdf)
        # st.write(text)  
        docs = Utils.split_doc(text, chunk_size=1000,chunk_overlap=100)
        # st.write(docs) 
        index = Utils.vector_Embedding(docs)

        # get similardoc
        query = st.text_area("Enter your question")
        button = st.button("Submit")
        
        if query and button:
            with st.spinner("Reading the Document..."):
                similar_docs = Utils.get_similar_docs(query, index)
                st.expander("Context").write(similar_docs)
                add_vertical_space(3)
                # Q&A using OpenAI
                answer = Utils.get_answer(query, similar_docs)
                st.write("Answer : "+ answer)

if __name__=="__main__":
    main()