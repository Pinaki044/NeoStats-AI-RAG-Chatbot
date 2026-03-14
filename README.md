# NeoStats AI RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot built as part of the NeoStats AI Engineer Use Case.

The chatbot answers user questions using information retrieved from documents and a large language model.

## Features

- Document-based question answering
- Retrieval using FAISS vector database
- Embeddings using HuggingFace sentence transformers
- LLM responses using Groq (Llama 3)
- Streamlit-based interactive chatbot UI

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM

## Project Structure
AI_UseCase
│
├── app.py (Streamlit chatbot interface)
├── config/ (API and configuration files)
├── documents/ (source PDFs for RAG)
├── models/ (LLM and embedding setup)
├── utils/ (retrieval and helper functions)
├── faiss_index/ (vector database)
└── requirements.txt


## Installation

Clone the repository:
git clone https://github.com/Pinaki044/NeoStats-AI-RAG-Chatbot.git


Install dependencies:
pip install -r requirements.txt


Set your Groq API key:
setx GROQ_API_KEY "your_api_key_here"


Run the application:
streamlit run app.py


## Usage

Open the Streamlit app and ask questions related to the uploaded documents.

The chatbot retrieves relevant document chunks and generates answers using the language model.

## Author

Pinaki Priya  
B.Tech CSE (Health Informatics)
