from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import glob

def build_index():
    pdf_files = glob.glob("./data/*.pdf")
    documents = []
    
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        documents.extend(loader.load())
    
    print(f"Loaded {len(documents)} pages")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./data/chroma"
    )
    print("Vectorstore built successfully!")

if __name__ == "__main__":
    build_index()

