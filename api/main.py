from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="STG Clinical RAG API")

rag_chain = None

def get_chain():
    global rag_chain
    if rag_chain is None:
        from rag.chain import get_rag_chain
        rag_chain = get_rag_chain()
    return rag_chain

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    chain = get_chain()
    return {"answer": chain(payload.question)}

@app.get("/health")
def health_check():
    return {"status": "ok"}