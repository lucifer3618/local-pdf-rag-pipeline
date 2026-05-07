from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_docs(self, docs):
        if not docs:
            logger.warning("No documents provided for chunking")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )
        all_chunks = []
        for doc in docs:
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc['id']}_{i}",
                    "doc_id": doc["id"],
                    "title": doc["title"],
                    "content": chunk,
                    "page_count": doc["page_count"],
                    "path": doc["path"]
                })

        if not all_chunks:
            logger.warning("Chunking produced no chunks")
            return []

        chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

        logger.info("Min chunk size: %s chars", min(chunk_lengths))
        logger.info("Max chunk size: %s chars", max(chunk_lengths))
        logger.info("Avg chunk size: %.0f chars", sum(chunk_lengths) / len(chunk_lengths))
        logger.info("Total chunks: %s", len(all_chunks))

        return all_chunks