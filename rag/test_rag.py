from rag.chain import get_rag_chain

rag = get_rag_chain()

question = "Treatment of severe malaria according to STG"
answer = rag(question)

print("\nANSWER:\n", answer)

