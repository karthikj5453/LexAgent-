from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import asyncio
import logging
from agents.supervisor import handle_request
from retrieval.chunker import chunk_document
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lexagent.api")

app = FastAPI(
    title="LexAgent API", 
    description="Multi-Agent Legal & Compliance Platform for Indian Enterprises",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrievers (will load existing indexes)
dense_retriever = DenseRetriever()
sparse_retriever = BM25Retriever()

# In-memory session store (use Redis/DB for production)
sessions: dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str
    doc_id: str | None = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF contract, chunk it page-by-page,
    and index it in both ChromaDB and BM25.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    doc_id = str(uuid.uuid4())
    os.makedirs("data/documents", exist_ok=True)
    file_path = f"data/documents/{doc_id}.pdf"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        logger.info(f"Saved uploaded contract: {file.filename} -> {file_path}")
        
        # Extract page-aware text chunks
        chunks = chunk_document(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text could be extracted from this PDF.")
            
        # Index in parallel (dense indexing is async, sparse indexing runs in a thread pool)
        await asyncio.gather(
            dense_retriever.index_chunks(chunks, doc_id),
            asyncio.to_thread(sparse_retriever.index_chunks, chunks, doc_id)
        )
        
        return {
            "doc_id": doc_id, 
            "filename": file.filename, 
            "chunks_indexed": len(chunks)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload and indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process and index document: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Primary chat query endpoint routed through the Supervisor agent.
    """
    session_id = request.session_id
    if session_id not in sessions:
        sessions[session_id] = []
    
    try:
        result = await handle_request(
            user_message=request.message,
            doc_id=request.doc_id,
            session_history=sessions[session_id],
        )
        
        # Update session history (in-memory)
        sessions[session_id].append({
            "role": "user", "content": request.message
        })
        sessions[session_id].append({
            "role": "assistant", "content": result["response"]
        })
        
        return result
    except Exception as e:
        logger.error(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Supervisor error: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model": settings.nim_model,
        "embedding_model": settings.nim_embedding_model
    }

# Entrypoint for running standard script directly
import asyncio
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
