import os
from groq import Groq
from rag.retriever import get_retriever

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def get_rag_chain():
    retriever = get_retriever()

    def rag_answer(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a clinical assistant using Tanzania STG. Give accurate, structured medical answers."},
                {"role": "user", "content": f"context:\n{context}\n\nQuestion:\n{question}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

    return rag_answer
