from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

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

@app.on_event("startup")
def startup_event():
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not set. API will return errors.")
    if not os.getenv("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY not set. API will return errors.")

@app.get("/health")
def health_check():
    status = "healthy" if os.getenv("GROQ_API_KEY") and os.getenv("GOOGLE_API_KEY") else "degraded"
    return {"status": status}

@app.post("/ask")
def ask_question(payload: QuestionRequest):
    try:
        chain = get_chain()
        answer = chain.invoke(payload.question)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")