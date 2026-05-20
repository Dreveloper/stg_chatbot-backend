cat > rag/retriever.py << 'EOF'
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

def get_retriever():
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory="./data/chroma",
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
EOF
