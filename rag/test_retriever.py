from retriever import get_retriever

query = "Management of malaria in pregnancy according to STG"

retriever = get_retriever()
docs = retriever.invoke(query)

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, 1):
    print(f"--- Document {i} ---")
    print(doc.page_content[:500])
    print()
