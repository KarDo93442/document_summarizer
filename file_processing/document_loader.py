from langchain.document_loaders import TextLoader

loader = TextLoader("./REAME.md")
loader.load()