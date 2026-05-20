from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

_vectorstore = None

def get_retriever():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        _vectorstore = Chroma(
            persist_directory="./data/chroma",
            embedding_function=embeddings
        )
    return _vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )