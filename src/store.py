import numpy as np
import faiss
from .embedder import Embedder
import chromadb

class StoreVectorDB:
    def __init__(self, embedder: Embedder, chunks):
        self.embedder = embedder
        self.chunks = chunks
        self.collection = self._create_chroma_index()

    def _create_chroma_index(self):
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_or_create_collection("rag_docs")

        if collection.count() == 0:
            print("Indexing chunks...")
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
            print(f"Indexed {collection.count()} chunks")
        else:
            print(f"Loaded {collection.count()} existing chunks")

        return collection
        
