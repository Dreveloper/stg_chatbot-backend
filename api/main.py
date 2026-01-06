from fastapi import FastAPI
from pydantic import BaseModel
from rag.chain import get_rag_chain

app = FastAPI(
    title="STG Clinical RAG API",
    description="Tanzania STG-based clinical decision suport (education only)",
    version="1.0"
)

rag = get_rag_chain()

class QuestionRequest(BaseModel):
      question: str

class AnswerResponse(BaseModel):
      answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    answer = rag(payload.question)
    return {"answer": answer}
