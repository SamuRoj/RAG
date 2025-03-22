# Build a Retrieval Augmented Generation 

This guide will show how to build a simple Q&A application over a text data source.

## Requirements

- Jupyter Notebook.
- LangChain.
- Pinecone Vector Database.
- OpenAI key.

## Architecture and Components

- **Language Model:** The core component that executes the prompts.
- **Vector Store:** Stores document embeddings for similarity search.
- **Indexing:** A pipeline for ingesting data from a source and indexing it.
- **Retrieval and generation:** The RAG chain that takes the user query at runtime and retrieves the relevant data from the index to pass it to the model.

## Installation

### LangChain Dependencies

- To install run:
```
pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
```

- Create a langsmith account to inspect the logs of the application, then import the following environment variables within the notebook and insert the langsmith api key when the input is prompted.

```
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Using Language Models

### OpenAI Setup

- Now we will use the language model of OpenAI, but first, we need to install it by running the following command, then enter the API key of OpenAI when the input is prompted and choose the model that will be used, in this example text-embedding-3-large is being used.

```
pip install -qU "langchain[openai]"
```

```
import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

### Pinecone Setup

- Execute the following command to install pinecone dependencies, create a new account in Pinecone and a new index to store the data that will be retrieved.
Replace <api_key> with the given one in Pinecone during the account creation and replace <index_name> with the name of the created index. 

```
pip install -qU langchain-pinecone
```

```
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=<api_key>)
index = pc.Index(<index_name>)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

## Usage

- Now replace <url> with the url of a webpage and the RAG will be able to answer questions of its content.
This script indexes the data, split it into the documents and saves it into pinecone. 

```
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("<url>",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
```

- Now we can ask any prompt about the webpage with the following command: 

```
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
```

![imagen](https://github.com/user-attachments/assets/37251a73-c28f-45ec-8867-0c225b0d76d0)

- To answer a different question, just change the value of the key question inside the dictionary.

```
response = graph.invoke({"question": "What is Task Self-Reflection?"})
print(response["answer"])
```

![imagen](https://github.com/user-attachments/assets/9766494e-a3b9-416c-8c22-cc5eb88a7bf0)

## Proof of the data inside the pinecone vector database

![imagen](https://github.com/user-attachments/assets/5206ef81-37a4-40c9-9e8b-2d8b0c6cfa76)

![imagen](https://github.com/user-attachments/assets/b580cd56-ca8d-4c5b-99d0-653bcfc033d1)

## Author

Samuel Rojas - SamuRoj
