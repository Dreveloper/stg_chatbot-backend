from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

_vectorstore = None

def get_retriever():
    global _vectorstore
    if _vectorstore is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        _vectorstore = Chroma(
            persist_directory="./data/chroma",
            embedding_function=embeddings
        )
    return _vectorstore.as_retriever(search_kwargs={"k": 3})