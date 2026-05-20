from fastapi import FastAPI
from pydantic import BaseModel
from rag.chain import get_rag_chain

app = FastAPI(title="STG Clinical RAG API")

rag_chain = get_rag_chain()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    answer = rag_chain(payload.question)
    return {"answer": answer}
