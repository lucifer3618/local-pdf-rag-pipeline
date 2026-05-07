import ollama

class Retriever:
    def __init__(self, model_name, embedder, chrom_collection, top_k=20):
        self.model_name = model_name
        self.embedder = embedder
        self.chrom_collection = chrom_collection
        self.top_k = top_k

    def _query_chrom(self, question):
        # Convert the question into an embedding
        query_embedding = self.embedder.embed(question).reshape(1, -1)
        print(f"Query embedding shape: {query_embedding.shape}")

        # Search the ChromaDB collection for the top 20 closest documents
        results = self.chrom_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"]
        )
        print("Distances:", results["distances"])
        print("Indices:", results["ids"])

        # Retrieve the documents based on the indices returned by ChromaDB
        retrieved_chunks = retrieved_chunks = results["documents"][0]
        docs = results["metadatas"][0]
        return retrieved_chunks, docs

    def query(self, question):
        retrieved_chunks, docs = self._query_chrom(question)
        # Create a prompt for the LLM using the retrieved documents and the question
        context = "\n".join(retrieved_chunks)
        
        prompt = f"""
            You are a research assistant specialized in analyzing academic papers.
            Your job is to answer questions accurately based ONLY on the provided research paper excerpts.

            STRICT RULES:
            - Only use information from the provided context
            - If the answer is not in the context, say "This information is not found in the provided papers"
            - Always cite which source you used [Source 1], [Source 2] etc.
            - Be precise and academic in tone
            - For numerical data (statistics, percentages) quote exactly as written
            - Do not make assumptions beyond what is stated

            CONTEXT FROM RESEARCH PAPERS:
            {context}

            QUESTION: {question}

            ANSWER (with citations):
        """
        response = ollama.chat(
        model=self.model_name,
        messages=[
            {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"], docs