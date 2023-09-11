import os
from typing import List

from tqdm import tqdm

from langchain.schema.document import Document
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from preprocess_configs import *


class DocumentLoader:
    def preprocess_file(filename: str) -> List[Document]:
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

    def process_knowledge_base(kb_filepath: str = KB_FILEPATH) -> List[Document]:
        pass
