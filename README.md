# Build a Retrieval Augmented Generation 

This guide will show how to build a simple Q&A application over a text data source.

## Requirements

- Jupyter Notebook
- LangChain
- Pynecone

### Installation Langchain

- To install run:
```
pip install --quiet --upgrade langchain-text-splitters langchain-community langgraph
```

- Create a langsmith account to inspect log out the results of the application, then import the following environment variables within the
notebook.

```
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

## Using Language models

- Now we will use the language model of openAI, but first, we need to install it by running the following command, enter the API key of OpenAI
when the input is prompted

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

- For the vector store Pynecone will be choosed, and it can be installed through the following command, a previous account in Pynecone is
needed

```
pip install -qU langchain-pinecone
```

```
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=...)
index = pc.Index(<index_name>)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

The API key has to be replaced with the given key that was created in Pinecone and an index has to be created, then replace <index_name> with
the name of the created index

## Usage

- Now the context that will be given is from a webpage and the RAG will be able to answer questions from the content of this page.
This scripts indexes the data, split it into the documents and save them into pynecone. 

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
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
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

## Proof of the data inside the pynecone vector database

![imagen](https://github.com/user-attachments/assets/5206ef81-37a4-40c9-9e8b-2d8b0c6cfa76)

![imagen](https://github.com/user-attachments/assets/b580cd56-ca8d-4c5b-99d0-653bcfc033d1)

## Author

Samuel Rojas - SamuRoj
