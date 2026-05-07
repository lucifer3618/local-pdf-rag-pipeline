import logging
from src.pipeline import RAGPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

if __name__ == "__main__":
    rag = RAGPipeline(
        docs_path="./pdfs",
        embedding_model="nomic-embed-text",
        chat_model="llama3.2:3b",
        chunk_size=500,
        overlap=50,
        top_k=10,
        chroma_path="./chroma_db",
        collection_name="rag_docs",
    )
    rag.initialize()

    question = "How many people worldwide affected by  snakebite envenoming in each year?"
    answer= rag.ask(question)
    print("Answer:", answer)