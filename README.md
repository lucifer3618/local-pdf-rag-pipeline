# Local PDF RAG Pipeline (Built from Scratch)

## Overview
This project is a localized Retrieval-Augmented Generation (RAG) pipeline designed specifically for querying academic research papers. It ingests local PDF documents, processes them, and provides a REST API to ask contextualized questions based *only* on the provided documents.

**Key Educational Focus:** This project was intentionally implemented **manually without relying on high-level orchestration frameworks** like LangChain (except for basic text splitting) or LlamaIndex. The core goal is to deeply understand the underlying architecture, data flow, and mechanics of the RAG concept, from raw document ingestion to LLM response generation.

## Why Build Manually?
Abstracting away the RAG process using large frameworks can obscure the core mechanisms at play. By building the pipeline manually, this project demonstrates:
- **Data Flow:** How raw PDF text is extracted, normalized, and managed.
- **Embedding Generation:** How text chunks are transformed into vector representations using a local embedding model API.
- **Vector Database Management:** How vectors and metadata are stored, indexed, and retrieved by interfacing directly with ChromaDB.
- **Prompt Engineering & Context Injection:** How retrieved chunks are injected into a strict, predefined LLM prompt to prevent hallucinations and enforce source tracking.
- **Local Inference Integration:** How to directly interface with local Ollama models for both embeddings and chat generation without intermediary wrappers.

## System Architecture

The pipeline is structured into discrete, single-purpose modules reflecting the true steps of a RAG system:

1. **Loader (`src/loader.py`):** Uses `pypdf` to read local PDF files from the `pdfs/` directory and extract raw text and metadata (like page counts).
2. **Chunker (`src/chunker.py`):** Splits the raw text into manageable, overlapping chunks to preserve contextual continuity.
3. **Embedder (`src/embedder.py`):** Interfaces with Ollama to generate vector embeddings for each chunk using the `nomic-embed-text` model.
4. **Vector Store (`src/store.py`):** Manages the connection to a local ChromaDB instance, storing the embeddings along with document metadata.
5. **Retriever (`src/retriver.py`):** Embeds the user's query, searches ChromaDB for the closest vector matches, and constructs a strict prompt with the retrieved context before querying the local LLM.
6. **API (`main.py`):** A FastAPI application that orchestrates the initialization of the pipeline on startup and exposes an endpoint for querying.

## Technologies Used
- **FastAPI:** High-performance web framework for the API.
- **Ollama:** Local LLM inference engine.
    - **Embeddings:** `nomic-embed-text`
    - **Chat/Generation:** `llama3.2:3b`
- **ChromaDB:** Open-source vector database.
- **PyPDF:** PDF processing library.
- **Pydantic:** Data validation.

## Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally.
- Required Ollama models pulled:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:3b
  ```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd rag-pipeline
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Documents:**
   Place your research paper PDFs in the `pdfs/` directory at the root of the project.

## Running the Application

### Option 1: Local Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
The application will initialize the RAG pipeline on startup (loading PDFs, chunking, embedding, and storing in ChromaDB). This may take some time depending on the number of PDFs.

### Option 2: Docker
You can also run the application using Docker Compose:
```bash
docker-compose up --build
```

## API Usage

The application exposes a single endpoint for querying the documents.

**Endpoint:** `POST /ask`

**Request Body:**
```json
{
  "question": "What is the main conclusion of the research?"
}
```

**Example cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What methodology was used?"
}'
```

**Response:**
```json
{
  "question": "What methodology was used?",
  "answer": "The researchers employed a qualitative approach... (Source: research_paper.pdf)"
}
```
