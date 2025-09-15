import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=FAISS.from_texts(text=text_chunks,embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    model_name = "meta-llama/Meta-LLama-3-8B-Instruct"
    pipe = pipeline("text-generation", model=model_name)
    llm = HuggingFacePipeline(pipeline=pipe)
       
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using the provided context."),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)

    retriever = vectorstore.as_retriever()
    conversation_chain = create_retrieval_chain(
        retriever=retriever,
        combine_documents_chain=combine_docs_chain
    )

    conversation_chain.memory = memory

    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({"question":user_question})
    st.write(response)
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            pass

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Financial Documents:")
    if "Conversation" not in st.session_state:
        st.session_state.conversation=None
    st.header("Chat with Multiple Financial Documents")
    st.text_input("Your questions for wealth")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your Finance PDFs",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
               raw_text=get_pdf_text(pdf_docs)
               st.write(raw_text)
               text_chunks=get_text_chunks(raw_text)
 
               vectorstore=get_vector_store(text_chunks)

               st.session_state.conversation=get_conversation_chain(vectorstore)
if __name__=="__main__":
    main()
