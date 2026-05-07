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
            # Fresh index
            self._index_chunks(collection, self.chunks)
        else:
            # Check if new chunks exist that aren't in DB yet
            existing_ids = set(collection.get()["ids"])
            new_chunks = [c for c in self.chunks if c["id"] not in existing_ids]

            if new_chunks:
                logger.info("Found %s new chunks to index", len(new_chunks))
                self._index_chunks(collection, new_chunks)
                logger.info("Added %s new chunks, total now %s", len(new_chunks), collection.count())
            else:
                logger.info("No new chunks - loaded %s existing", collection.count())

        return collection

    def _index_chunks(self, collection, chunks):
        logger.info("Indexing %s chunks into %s", len(chunks), self.collection_name)
        embeddings = [self.embedder.embed(c["content"]) for c in chunks]
        collection.add(
            documents=[c["content"] for c in chunks],
            embeddings=embeddings,
            ids=[c["id"] for c in chunks],
            metadatas=[{
                "doc_id":     c["doc_id"],
                "title":      c["title"],
                "page_count": c["page_count"],
                "path":       c["path"]
            } for c in chunks]
        )
        logger.info("Indexed %s chunks", collection.count())
        
