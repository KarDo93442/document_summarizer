from typing import List

from transformers import pipeline, Pipeline

from langchain.vectorstores.base import VectorStoreRetriever

from llm_configs import LLM_NAME, LLM_FOR_EXTRACTIVE_QA


class LLMforExtractingQA:
    def __init__(self, vs_retriever: VectorStoreRetriever, model_name: str = LLM_NAME):
        self.retriever = vs_retriever
        model: Pipeline = pipeline(task=LLM_FOR_EXTRACTIVE_QA, model=model_name)
        self.llm = model

    def get_llm_answer(self, question: str, context: str or List[str]) -> str:
        if isinstance(context, list):
            text = ""
            for e in context:
                text += e

        result = self.llm(question=question, context=context)
        return result["answer"]
