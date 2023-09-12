from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

from preprocess_configs import EMBEDDING_MODEL, EMBEDDING_DEVICE, CACHE_FOLDER


class DocumentEmbedder:
    def create_huggingface_embeddings(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_device: str = EMBEDDING_DEVICE,
        cache_folder: str = CACHE_FOLDER,
    ) -> Embeddings:
        embeddings: Embeddings = HuggingFaceEmbeddings(
            embedding_model,
            model_kwargs={"device": embedding_device},
            cache_folder=cache_folder,
        )
        return embeddings
