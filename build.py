import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai 
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

load_dotenv()

class KB:
    def __init__(self):
        genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

    def extract_pdf_text(self,pdfs):
        text = ""
        for pdf in pdfs:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = text + page.extract_text()
        return text
    
    def extract_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def store_vectors(self,text_chunks,faiss_path = "FAISS_store"):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        faiss_store = FAISS.from_texts(text_chunks,embedding = embeddings)
        faiss_store.save_local(faiss_path)


class KB_Front_End:
    def run(self):
        ingest=KB()
        st.set_page_config("LLM Avatar")
        st.header("LLM Avatar Knowledge base Creation")
        pdfs = st.file_uploader("Upload PDfs",accept_multiple_files=True)
        if st.button("Store Knowledge"):
            with st.spinner("Building the Knowledge Base"):
                try:
                    raw_text = ingest.extract_pdf_text(pdfs)
                    text_chunks = ingest.extract_text_chunks(raw_text)
                    ingest.store_vectors(text_chunks)

                except Exception as e:
                    st.exception(f"Exception: {e}")
                st.success("Created")
if __name__=="__main__":
    FE = KB_Front_End()
    FE.run()



        