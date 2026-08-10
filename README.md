# Local RAG Technical Support Assistant

A local question-answering assistant built with Microsoft Foundry Local, SQLite, and the Retrieval-Augmented Generation (RAG) pattern.

The application retrieves relevant information from local documents and generates grounded answers without requiring a cloud API.

## Features

- Runs AI models locally with Microsoft Foundry Local
- Stores document chunks and embeddings in SQLite
- Uses semantic search and cosine similarity
- Generates answers using retrieved context
- Displays retrieved sources and similarity scores
- Refuses to answer when the knowledge base is insufficient
- Works offline after the models are downloaded

## Technologies

- Python 3.13
- Microsoft Foundry Local
- SQLite
- Qwen3 Embedding 0.6B
- Phi-3.5 Mini
- Cosine similarity

## Installation

Create and activate a virtual environment:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bat
pip install -r requirements.txt
```

## Prepare the Knowledge Base

Generate embeddings and save the document chunks to SQLite:

```bat
python ingest.py
```

## Run the Application

```bat
python main.py
```

Type `quit` to close the application.

## Example Questions

```text
How does RAG reduce hallucinations?
Which database is used for local storage?
What are embeddings?
Who founded Microsoft?
```

The final question cannot be answered using the provided documents, so the assistant should state that the knowledge base does not contain enough information.

## How It Works

1. Text files are read from the `documents` directory.
2. Documents are divided into smaller chunks.
3. The Qwen3 embedding model converts each chunk into a numerical vector.
4. Document chunks and vectors are stored in SQLite.
5. The user's question is converted into an embedding.
6. Cosine similarity finds the most relevant document chunks.
7. The retrieved context is sent to the local language model.
8. The model generates an answer using only the retrieved information.

## Project Structure

- `documents/`: Local knowledge-base documents
- `database.py`: SQLite database operations
- `ingest.py`: Document ingestion and embedding generation
- `main.py`: Interactive RAG application
- `embedding_test.py`: Semantic-search test
- `requirements.txt`: Python dependencies

## Limitations

- The current knowledge base contains a small document collection.
- Similarity search compares vectors in Python and is intended for small datasets.
- Answer quality depends on the selected local language model.
- The current interface is command-line based.