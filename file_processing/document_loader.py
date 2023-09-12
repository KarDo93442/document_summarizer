import os
import sys
from typing import List

from tqdm import tqdm

from langchain.schema.document import Document
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from preprocess_configs import KB_FILEPATH, separators, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoader:
    def preprocess_file(self, filename: str) -> List[Document]:
        if filename.lower().endswith(".txt"):
            loader = TextLoader(file_path=filename)
            textsplitter = CharacterTextSplitter(
                separator=separators, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs: List[Document] = loader.load_and_split(text_splitter=textsplitter)

        elif filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path=filename)
            textsplitter = CharacterTextSplitter(
                separator=separators, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = loader.load_and_split(text_splitter=textsplitter)

        return docs

    def process_knowledge_base(self, kb_filepath: str = KB_FILEPATH) -> List[Document]:
        if not os.path.exists(kb_filepath):
            print(f"This knowledge base path doesn't exist\n")
            sys.exit(-1)

        if os.path.isfile(kb_filepath):
            docs: List[Document] = self.preprocess_file(kb_filepath)

        elif os.path.isdir(kb_filepath):
            docs = []

            for file in tqdm(os.listdir(kb_filepath), desc="Loading files..."):
                filepath = os.path.join(kb_filepath, file)
                docs += self.preprocess_file(filename=filepath)

        return docs
