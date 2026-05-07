from .embedder import Embedder
import chromadb
import logging

logger = logging.getLogger(__name__)

class StoreVectorDB:
    def __init__(self, embedder: Embedder, chunks, db_path="./chroma_db", collection_name="rag_docs"):
        self.embedder = embedder
        self.chunks = chunks
        self.db_path = db_path
        self.collection_name = collection_name
        self.collection = self._create_chroma_index()

    def _create_chroma_index(self):
        client = chromadb.PersistentClient(path=self.db_path)
        collection = client.get_or_create_collection(self.collection_name)

        if collection.count() == 0:
            if not self.chunks:
                raise ValueError("Cannot create a new vector index with empty chunks")

            logger.info("Indexing %s chunks into %s", len(self.chunks), self.collection_name)
            embeddings = [self.embedder.embed(c["content"]) for c in self.chunks]
            collection.add(
                documents=[c["content"] for c in self.chunks],
                embeddings=embeddings,
                ids=[c["id"] for c in self.chunks],
                metadatas=[{
                    "doc_id": c["doc_id"], 
                    "title": c["title"],
                    "page_count": c["page_count"],
                    "path": c["path"]
                    } for c in self.chunks]
            )
            logger.info("Indexed %s chunks", collection.count())
        else:
            logger.info("Loaded %s existing chunks", collection.count())

        return collection
        
