from typing import List

from langchain.schema.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.base import VectorStoreRetriever


class VectorDB:
    def __init__(self, chunked_docs: List[Document], embeddings: Embeddings):
        self.docs = chunked_docs
        self.embedding = embeddings

    def create_chroma_vectorstore(self) -> VectorStore:
        vectordb: VectorStore = Chroma.from_documents(
            documents=self.docs, embedding=self.embedding
        )
        return vectordb

    def get_db_retriever(self) -> VectorStoreRetriever:
        vectordb = self.create_chroma_vectorstore()
        return vectordb.as_retriever()
