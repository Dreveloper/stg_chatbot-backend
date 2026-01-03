from rag.retriever import get_retriever
from rag.prompt import STG_PROMPT
from langchain_community.llms import HuggingFacePipeline


def get_rag_chain():
    retriever = get_retriever()

    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={
            "max_length": 512
        },
    )

    def rag_answer(question: str):
        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = STG_PROMPT.format(
            context=context,
            question=question
        )

        return llm.invoke(prompt)

    return rag_answer

