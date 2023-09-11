# Document summarizer

## Project description
The system inputs a user's question related to the system's knowledge base and outputs the short and consice answer. To initiate the knowledge base, system takes text documents, processes them and creates embeddings from them. These embeddings are stored in a vector database, which allows for similarity search. Thanks to it, the LLM can get the short relevant information from kowledge base and output consised answer to the user's question.

### System features:
* supports *txt*, *pdf* file formats
* available from  jupyter notebook, (coming soon) cli and GUI
* can store knowledge base on the disk (coming soon)

## Purpose
The goal of this project is to simplify the process of finding and extracting information from text documents. Main intend of the procject for me, the author, is to accelarate learning and save time in collage.


## Installation guide
It is recommended to create a virtual environment to run this project.

For conda: 
```conda create -n document_summarizer python pip```

For python: 
```python -m venv /path/to/the/environment```

Then install the required libraries by running
```pip install -r requirements.txt```


## Usage instructions

### For jupyter (example results without running the program)

### For cli

### For webapp



## License
* Python 3.11.4
* Langchain 0.0.285

