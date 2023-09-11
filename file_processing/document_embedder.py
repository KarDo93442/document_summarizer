from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings


class DocumentEmbedder:
    def create_huggingface_embeddings(
        embedding_model: str, embedding_device: str, cache_folder: str
    ) -> Embeddings:
        embeddings: Embeddings = HuggingFaceEmbeddings(
            embedding_model,
            model_kwargs={"device": embedding_model},
            cache_folder=cache_folder,
        )
        return embeddings
