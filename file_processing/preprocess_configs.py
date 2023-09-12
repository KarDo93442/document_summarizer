import torch.cuda
import torch.backends

KB_FILEPATH = "./README.md"
separators = ["\n\n", "\n", ",", ".", "!", "?"]
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
CACHE_FOLDER = "~/.cache/huggingface/"
