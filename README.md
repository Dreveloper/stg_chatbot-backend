# STG Clinical RAG API

A Tanzania STG–based clinical decision support system (education only)
built using Retrieval-Augmented Generation (RAG).

⚠️ DISCLAIMER:
This project is for educational purposes only.
It is NOT a substitute for clinical judgment or official guidelines.

## Project Structure
stg_chatbot/
- ingestion/   # PDF ingestion and FAISS index creation
- rag/         # Retriever and RAG chain logic
- api/         # FastAPI application
- data/
  - raw/       # STG PDF (not pushed to git)
  - vectors/   # FAISS index (generated locally)

## Setup
1. Create virtual environment
   python -m venv venv
   source venv/bin/activate

2. Install dependencies
   pip install -r requirements.txt

3. Build vector index
   python ingestion/build_stg_index.py

4. Run API
   uvicorn api.main:app --reload

Open:
http://127.0.0.1:8000/docs

## Example Request
POST /ask

{
  "question": "malaria treatment in adults"
}

