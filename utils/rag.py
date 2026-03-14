import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from models.embeddings import get_embedding_model

vectorstore = None


def load_documents():
    """Load all PDFs from documents folder"""
    try:
        documents = []
        folder_path = "documents"

        for file in os.listdir(folder_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(folder_path, file))
                documents.extend(loader.load())

        return documents

    except Exception as e:
        raise RuntimeError(f"Document loading error: {str(e)}")


def create_vectorstore():
    """Create or load FAISS vector database"""
    global vectorstore

    try:
        embeddings = get_embedding_model()

        # If FAISS index exists, load it
        if os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )

        # Otherwise create new one
        else:
            docs = load_documents()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(docs)

            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Save the FAISS index
            vectorstore.save_local("faiss_index")

        return vectorstore

    except Exception as e:
        raise RuntimeError(f"Vector store error: {str(e)}")


def retrieve_context(query):
    """Retrieve relevant document chunks"""
    try:
        vectorstore = create_vectorstore()

        docs = vectorstore.similarity_search(query, k=3)

        context = "\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

        return context, sources

        return context

    except Exception as e:
        return f"Retrieval error: {str(e)}"