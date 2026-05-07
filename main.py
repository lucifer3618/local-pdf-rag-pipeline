from src.loader import PDFLoader
from src.chunker import Chuncker
from src.embedder import Embedder
from src.retriver import Retriever
from src.store import StoreVectorDB

if __name__ == "__main__":
    # Load documents
    loader = PDFLoader()
    docs = loader.load("./pdfs")

    # Chunk documents
    chunker = Chuncker(chunk_size=500, overlap=50)
    chunks = chunker.chunk_docs(docs)

    # Create embedder and store
    embedder = Embedder("nomic-embed-text")
    store = StoreVectorDB(embedder, chunks)
    
    # Create retriever and query
    retriever = Retriever("llama3.2:3b", embedder, store.collection, top_k=10)
    question = "How many people worldwide affected by  snakebite envenoming in each year?"
    answer, docs_n = retriever.query(question)
    print("Retrieved docs:", docs_n)
    print("Answer:", answer)