from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from agents.supervisor import handle_request
from retrieval.chunker import chunk_document
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from config import settings
from api.database import init_db, get_db, Document as DBDocument, ChatSession, Message as DBMessage
from api.tracing import setup_tracing

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("lexagent.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and tracing hooks
    logger.info("Initializing database tables...")
    init_db()
    logger.info("Setting up observability tracing...")
    setup_tracing()
    yield
    # Shutdown logic (if any)

app = FastAPI(
    title="LexAgent API", 
    description="Multi-Agent Legal & Compliance Platform for Indian Enterprises",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database tables
init_db()

# Initialize retrievers (will load existing indexes)
dense_retriever = DenseRetriever()
sparse_retriever = BM25Retriever()


class ChatRequest(BaseModel):
    message: str
    session_id: str
    doc_id: str | None = None

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        
        # Save document metadata to the database
        db_doc = DBDocument(
            doc_id=doc_id,
            filename=file.filename,
            filepath=file_path,
            chunks_indexed=len(chunks)
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        
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
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Primary chat query endpoint routed through the Supervisor agent.
    """
    session_id = request.session_id
    
    # Retrieve or create session in DB
    db_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not db_session:
        db_session = ChatSession(session_id=session_id)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
    
    # Retrieve chronological session history
    db_messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.id.asc()).all()
    session_history = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    
    try:
        result = await handle_request(
            user_message=request.message,
            doc_id=request.doc_id,
            session_history=session_history,
        )
        
        # Append user message and assistant response to database
        user_msg = DBMessage(session_id=session_id, role="user", content=request.message)
        assistant_msg = DBMessage(session_id=session_id, role="assistant", content=result["response"])
        db.add(user_msg)
        db.add(assistant_msg)
        db.commit()
        
        return result
    except Exception as e:
        logger.error(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Supervisor error: {str(e)}")

@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    """Retrieve all uploaded documents."""
    docs = db.query(DBDocument).order_by(DBDocument.created_at.desc()).all()
    return [{"doc_id": d.doc_id, "filename": d.filename, "chunks_indexed": d.chunks_indexed, "created_at": d.created_at} for d in docs]

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve the chat history of a specific session."""
    messages = db.query(DBMessage).filter(DBMessage.session_id == session_id).order_by(DBMessage.id.asc()).all()
    return [{"role": m.role, "content": m.content, "timestamp": m.timestamp} for m in messages]

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model": settings.nim_model,
        "embedding_model": settings.nim_embedding_model
    }

# Entrypoint for running standard script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)

