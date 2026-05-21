from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import get_retriever
import os

_rag_chain = None

def get_rag_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    retriever = get_retriever()
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clinical assistant using Tanzania STG (Standard Treatment Guidelines). Give accurate, structured medical answers based on the provided context."),
        ("user", "Context:\n{context}\n\nQuestion:\n{question}")
    ])
    
    _rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return _rag_chain