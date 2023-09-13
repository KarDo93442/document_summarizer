import os
import sys
from typing import List

from langchain.schema.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.base import VectorStoreRetriever

from db_configs import DB_PATH


class VectorDB:
    def __init__(self, chunked_docs: List[Document], embeddings: Embeddings):
        self.docs = chunked_docs
        self.embedding = embeddings

    def create_chroma_vectorstore(
        self, db_path: str = None, save_db_path: str = DB_PATH
    ) -> VectorStore:
        if db_path is None:
            vectordb: VectorStore = Chroma.from_documents(
                documents=self.docs, embedding=self.embedding, persist_directory=db_path
            )
            return vectordb

        if not os.path.exists(db_path):
            print(f"The path to chroma db - {db_path}, does not exist\n")
            sys.exit(-2)

        vectordb: VectorStore = Chroma(
            persist_directory=db_path, embedding_function=self.embedding
        )
        return vectordb

    def get_db_retriever(self) -> VectorStoreRetriever:
        vectordb: VectorStore = self.create_chroma_vectorstore()
        return vectordb.as_retriever()
