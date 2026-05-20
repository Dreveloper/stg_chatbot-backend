from rag.chain import get_rag_chain

rag = get_rag_chain()

question = "clinical features of malaria"
answer = rag(question)

print("\nANSWER:\n", answer)

