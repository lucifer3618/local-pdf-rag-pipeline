from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel, Field
from src.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs ONCE at startup
    logger.info("Initializing RAG pipeline...")
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

    # Store the RAG pipeline in app state for access in endpoints
    app.state.rag = rag
    logger.info("RAG pipeline ready!")
    yield
    # Runs ONCE at shutdown
    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)


# ---- Pydantic models for request and response ----
class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="The question to ask the RAG system"
    )

class QueryResponse(BaseModel):
    question: str
    answer: str


# ---- API Endpoints ----
@app.get("/")    
async def root():
    return {"message": "Welcome to the RAG API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    # Extract the RAG pipeline from app state
    rag = app.state.rag
    answer = rag.ask(query.question)
    return QueryResponse(question=query.question, answer=answer)