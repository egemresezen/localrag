# LocalDoc AI

A fully local Retrieval-Augmented Generation (RAG) application for asking questions about your own documents.

LocalDoc AI runs its language and embedding models with Microsoft Foundry Local, stores document chunks in SQLite, and provides a Streamlit interface. After the required models are downloaded, document processing and question answering can run locally without a cloud API.

## Demo

[Watch the LocalDoc AI demo video on Google Drive](https://drive.google.com/file/d/1IJI638Bv6k7rmfMSZJ0zgOqCGCQ8ILFL/view?usp=drive_link)

## Features

- Upload multiple PDF, DOCX, and TXT documents
- Extract text from document paragraphs and DOCX tables
- Split documents into overlapping chunks
- Generate embeddings in small batches for improved reliability
- Store document chunks and embeddings in SQLite
- Retrieve the most relevant chunks with cosine similarity
- Generate answers using only the retrieved document context
- Display sources, retrieved chunks, and similarity scores
- Delete individual documents from the knowledge base
- Clear chat history from the interface
- Run locally without a cloud API after model setup

## Supported File Types

| Format | Support | Notes |
| --- | --- | --- |
| TXT | Yes | UTF-8 and Latin-1 text files |
| PDF | Yes | Text-based PDFs; scanned PDFs require OCR |
| DOCX | Yes | Paragraphs and table contents are supported |

## Technologies

- Python 3.13
- Microsoft Foundry Local
- Streamlit
- SQLite
- Qwen3 Embedding 0.6B
- Phi-3.5 Mini
- PyPDF
- python-docx
- Cosine similarity

## How It Works

1. The user uploads one or more PDF, DOCX, or TXT documents.
2. Text is extracted, cleaned, and divided into overlapping chunks.
3. Qwen3 Embedding converts each chunk into a numerical vector.
4. Chunks, source names, and embeddings are stored in SQLite.
5. The user's question is converted into an embedding.
6. Cosine similarity ranks the stored chunks by relevance.
7. The most relevant chunks are passed to Phi-3.5 Mini as context.
8. The model answers using only the retrieved context.
9. The interface displays the answer together with its sources and similarity scores.

## Requirements

- Windows
- Python 3.13
- Git
- Internet access for the initial model download
- Sufficient free memory and disk space for the local models

## Installation

Clone the repository:

```bat
git clone https://github.com/egemresezen/localrag.git
cd localrag
```

Create and activate a virtual environment:

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit interface:

```bat
streamlit run app.py
```

If the browser does not open automatically, visit:

```text
http://localhost:8501
```

The first start may take longer because Foundry Local checks, downloads, and loads the selected models.

## Usage

1. Open the **Knowledge Base** tab.
2. Select one or more PDF, DOCX, or TXT files.
3. Click **Add to Knowledge Base** and wait for indexing to finish.
4. Open the **Chat** tab.
5. Ask a question about the uploaded documents.
6. Expand **Retrieved sources** to inspect the source text and similarity scores.
7. Test grounding by asking a question whose answer is not present in the documents.

## Project Structure

```text
localrag/
|-- app.py                 # Streamlit user interface
|-- rag_service.py         # Model loading, retrieval, and answer generation
|-- document_processor.py  # PDF, DOCX, and TXT extraction and chunking
|-- database.py            # SQLite storage operations
|-- requirements.txt       # Python dependencies
`-- README.md              # Project documentation
```

The `rag.db` SQLite database is created locally and stores the indexed document chunks and embeddings.

## Grounded Answering

The system prompt instructs the language model to answer only from retrieved document context. When the required information is unavailable, the expected response is:

```text
I do not have enough information in the knowledge base.
```

This behavior helps reduce unsupported answers and makes the result easier to verify through the displayed sources.

## Limitations

- Scanned or image-only PDFs require OCR, which is not included.
- CPU-based local inference can take longer than cloud-based inference.
- Similarity search is performed in Python and is intended for small or medium local knowledge bases.
- Answer quality depends on the selected models and the quality of the uploaded documents.
- Retrieved context reduces hallucinations but cannot guarantee that every generated answer is correct.

## Privacy

Documents, embeddings, the SQLite database, retrieval, and answer generation remain on the local machine. A cloud API key is not required.

## Author

Developed by [Ege Emre Sezen](https://github.com/egemresezen).
