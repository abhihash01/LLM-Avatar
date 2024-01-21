import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
faiss_path = "FAISS_store"
class prompt_engineering:

    def get_prompt_template(self):
        prompt_template="""
        You are a person Abhilash. The context here has information from Abhilash's resume.\
        Provide answer to the question from the given context in a detailed format. If the answer \
        is not available in the context, say "I don't have the answer to this in my KB. \
        Why don't you reach out to me in person/ by email / by call so I can provide a detailed answer?"

        Context: \n {context}? \n
        Question: \n {question} \n

        Answer:

        """
        prompt = PromptTemplate(template=prompt_template,input_variables = ["context","question"])
        return prompt
    
class conversatoinal_chain:
    
    def __init__(self):
        self.chat_model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.5)

    def load_chat(self):
        return self.chat_model

    def load_chain(self):
        prompter = prompt_engineering()
        prompts = prompter.get_prompt_template()
        chain = load_qa_chain(self.chat_model,chain_type="stuff",prompt=prompts)   
        return chain

class input_processor:
    def __init__(self):
        self.gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def get_similar_docs(self,question,faiss_path = faiss_path):
        faiss_db = FAISS.load_local(faiss_path,self.gemini_embeddings)
        docs = faiss_db.similarity_search(question)
        return docs
    
    def get_knowledge_response(self,chain,docs,question):
        response = chain({"input_documents": docs, "question": question})
        return response


class application:

    def run(self):
        question = st.text_input("Ask Me About Myself")

        submit1 = st.button("Ask Me")

        conver_chain = conversatoinal_chain()

        inp_processor = input_processor()

        response = []

        if submit1:
            #chat = conver_chain.load_chat()
            #response = inp_processor.get_response(chat,question)
            #st.write("Response: ", response.content)
            if question:
                chain = conver_chain.load_chain()
                docs = inp_processor.get_similar_docs(question)
                response = inp_processor.get_knowledge_response(chain,docs,question)
                st.write("Response: ", response["output_text"])
            else:
                st.write("Response: The question field is empty")
        return

if __name__ == "__main__":
    application=application()
    application.run()