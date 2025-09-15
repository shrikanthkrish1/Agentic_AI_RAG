Agentic AI

This project is a Streamlit web app that lets you upload multiple PDF files (like annual reports, statements, or research papers) and then ask natural-language questions about their content. Under the hood, it uses LangChain, Hugging Face embeddings, and a retrieval-augmented generation pipeline to pull relevant information from your PDFs and generate answers.

✨ Features

Upload one or more financial PDFs directly in the sidebar.

Extract and split the text into manageable chunks for efficient retrieval.

Store document chunks in a FAISS vector database using hkunlp/instructor-xl embeddings.

Ask questions in plain English and get answers based on your uploaded documents.

Keeps a simple conversation memory so it can handle follow-up questions.

🛠️ How it Works

PDF Text Extraction
Each uploaded PDF is read with PyPDF2.PdfReader and the text is extracted.

Chunking
Text is split into overlapping chunks (1000 characters, 300 overlap) with CharacterTextSplitter to improve retrieval quality.

Vector Store Creation
Each chunk is converted into an embedding using Hugging Face’s hkunlp/instructor-xl model.
The embeddings are stored in a local FAISS index for fast similarity search.

Question Answering
A lightweight meta-llama/Meta-LLaMA-3-8B-Instruct model is wrapped in a Hugging Face pipeline for generation.
LangChain combines document retrieval with a simple prompt template: “Answer the question using the provided context.”

Streamlit UI

Users upload PDFs and type questions through the browser. Processing and answers happen in real time.
Users upload PDFs and type questions through the browser. Processing and answers happen in real time.
