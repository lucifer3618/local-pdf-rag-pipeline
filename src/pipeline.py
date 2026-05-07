import logging
from typing import Any

from src.loader import PDFLoader
from src.chunker import Chunker
from src.embedder import Embedder
from src.store import StoreVectorDB
from src.retriver import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        docs_path: str = "./pdfs",
        embedding_model: str = "nomic-embed-text",
        chat_model: str = "llama3.2:3b",
        chunk_size: int = 500,
        overlap: int = 50,
        top_k: int = 10,
        chroma_path: str = "./chroma_db",
        collection_name: str = "rag_docs",
    ) -> None:
        self.docs_path = docs_path
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.chroma_path = chroma_path
        self.collection_name = collection_name

        self._retriever: Retriever | None = None

    def initialize(self) -> None:
        loader = PDFLoader()
        docs = loader.load(self.docs_path)
        if not docs:
            raise RuntimeError(f"No PDF documents were loaded from {self.docs_path}")

        chunker = Chunker(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks = chunker.chunk_docs(docs)
        if not chunks:
            raise RuntimeError("No chunks were produced from loaded documents")

        embedder = Embedder(self.embedding_model)
        store = StoreVectorDB(
            embedder,
            chunks,
            db_path=self.chroma_path,
            collection_name=self.collection_name,
        )
        self._retriever = Retriever(self.chat_model, embedder, store.collection, top_k=self.top_k)
        logger.info("RAG pipeline initialized")

    def ask(self, question: str) -> str:
        if self._retriever is None:
            raise RuntimeError("RAG pipeline not initialized. Call initialize() first.")

        answer = self._retriever.query(question)
        return answer
