from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chuncker:
    def __init__(self, chunk_size=500, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_docs(self, docs):
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

        chunk_lengths = [len(chunk["content"]) for chunk in all_chunks]

        print(f"Min chunk size: {min(chunk_lengths)} chars")
        print(f"Max chunk size: {max(chunk_lengths)} chars")
        print(f"Avg chunk size: {sum(chunk_lengths)/len(chunk_lengths):.0f} chars")
        print(f"Total chunks: {len(all_chunks)}")

        return all_chunks