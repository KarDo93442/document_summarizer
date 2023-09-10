from typing import List

from langchain.schema.document import Document
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from preprocess_configs import *


class DocumentLoader:
    def preprocess_file(filename: str) -> List[Document]:
        loader = TextLoader(KB_FILEPATH)
        textsplitter = CharacterTextSplitter(
            separator=separators, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs: List[Document] = loader.load_and_split(textsplitter)

        return docs

    def process_knowledge_base(kb_filepath: str = KB_FILEPATH) -> List[Document]:
        pass
