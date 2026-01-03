from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def get_llm():
    pipe = pipeline(
        task="text2text-generation",
        model ="google/flan-t5-base",
        max_new_tokens=256
)   

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
