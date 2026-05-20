from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

_vectorstore = None

def get_retriever():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory="./data/chroma",
            embedding_function=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
        )
    return _vectorstore.as_retriever(search_kwargs={"k": 3})