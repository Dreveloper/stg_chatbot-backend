from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

_vectorstore = None

def get_retriever():
    global _vectorstore
    if _vectorstore is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        _vectorstore = Chroma(
            persist_directory="./data/chroma",
            embedding_function=embeddings
        )
    return _vectorstore.as_retriever(search_kwargs={"k": 3})