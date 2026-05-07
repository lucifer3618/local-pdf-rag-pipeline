import ollama
import logging

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    pass


class Retriever:
    def __init__(self, model_name, embedder, chrom_collection, top_k=20):
        self.model_name = model_name
        self.embedder = embedder
        self.chrom_collection = chrom_collection
        self.top_k = max(1, int(top_k))

    def _query_chrom(self, question):
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string")

        # Convert the question into an embedding
        query_embedding = self.embedder.embed(question).reshape(1, -1)
        logger.debug("Query embedding shape: %s", query_embedding.shape)

        # Search the ChromaDB collection for the top 20 closest documents
        results = self.chrom_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        logger.debug("Distances: %s", results.get("distances"))
        logger.debug("Indices: %s", results.get("ids"))

        # Retrieve the documents based on the indices returned by ChromaDB
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        retrieved_chunks = documents[0] if documents else []
        docs = metadatas[0] if metadatas else []

        if not retrieved_chunks:
            raise RetrievalError("No relevant context retrieved from vector store")

        return retrieved_chunks, docs

    def query(self, question):
        retrieved_chunks, docs = self._query_chrom(question)
        # Create a prompt for the LLM using the retrieved documents and the question
        context = "\n".join(retrieved_chunks)
        
        prompt = f"""
            You are a research assistant specialized in analyzing academic papers.
            Your job is to answer questions accurately based ONLY on the provided 
            research paper excerpts below.

            STRICT RULES:
            - Only use information from the provided context
            - If the answer is not in the context, say "This information is not 
            found in the provided papers"
            - CRITICAL: The context contains numbers in square brackets like [1], 
            [4], [26], [31] etc. These are bibliography references INSIDE the 
            papers. You MUST IGNORE these completely. Never include them in 
            your answer under any circumstances.
            - End your answer with (Source: filename.pdf) using the paper filename
            - Be precise and academic in tone
            - For numerical data quote exactly as written
            - Do not make assumptions beyond what is stated

            CONTEXT FROM RESEARCH PAPERS:
            {context}

            QUESTION: {question}

            ANSWER (no bracket numbers, end with Source: filename.pdf):
        """
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = response.get("message", {}).get("content")
        if not content:
            raise RuntimeError("LLM response did not contain content")

        return content