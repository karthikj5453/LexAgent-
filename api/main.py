import sys
import os

# Auto-configure python path for direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
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

@app.post("/demo/load")
async def load_demo_data(db: Session = Depends(get_db)):
    """
    Seeds the database with a high-fidelity sample contract 
    and mock agent results to enable immediate exploration.
    """
    demo_doc_id = "demo-contract-india-nda"
    session_id = "session_hackathon_demo"
    
    # 1. Check if demo doc already exists
    existing_doc = db.query(DBDocument).filter(DBDocument.doc_id == demo_doc_id).first()
    if existing_doc:
        # Clear out messages to refresh session
        db.query(DBMessage).filter(DBMessage.session_id == session_id).delete()
        db.commit()
    else:
        # Create document entry
        db_doc = DBDocument(
            doc_id=demo_doc_id,
            filename="Sample_NDA_Indian_Enterprise.pdf",
            filepath="data/documents/sample_nda.pdf",
            chunks_indexed=12
        )
        db.add(db_doc)
        
    # 2. Add sample chat history
    demo_msgs = [
        DBMessage(session_id=session_id, role="user", content="Review this NDA and check if it complies with Indian law."),
        DBMessage(session_id=session_id, role="assistant", content="### LexAgent Analysis Report\n\nI have completed the review of **Sample_NDA_Indian_Enterprise.pdf** under the Indian Contract Act, 1872, and the Information Technology Act, 2000.\n\n* **Risk Profile**: MEDIUM\n* **Jurisdiction**: Out of State (Delhi Courts specified; should be Mumbai/Karnataka based on parties).\n* **Key Issues**: Section 27 Restraint of Trade restriction flagged in the confidentiality clause.")
    ]
    for msg in demo_msgs:
        db.add(msg)
        
    db.commit()
    
    # 3. Return pre-made mock results that the frontend can read immediately
    mock_results = {
        "doc_id": demo_doc_id,
        "filename": "Sample_NDA_Indian_Enterprise.pdf",
        "chunks_indexed": 12,
        "agent_results": {
            "review": {
                "document_type": "Non-Disclosure Agreement (NDA)",
                "parties": ["Acme Enterprises Pvt Ltd", "Karthik Consulting Services"],
                "effective_date": "2026-06-25",
                "overall_risk": "medium",
                "executive_summary": "This is a mutual non-disclosure agreement. It is generally standard but contains an overly broad restraint of trade clause under Section 27 of the Indian Contract Act and lacks clear governing law provisions for e-signatures.",
                "clauses": [
                    {
                        "clause_type": "Restraint of Trade / Non-Compete",
                        "clause_text": "The receiving party shall not engage in any competing consulting services in India for a period of 3 years post termination.",
                        "page_reference": "Page 3, Section 8.2",
                        "risk_level": "critical",
                        "risk_explanation": "Direct restraint of trade. Section 27 of the Indian Contract Act, 1872 states that any agreement in restraint of trade is void to that extent. This clause is completely unenforceable in Indian courts.",
                        "suggested_revision": "The receiving party shall not use any Confidential Information of the disclosing party to solicit the disclosing party's active clients for a period of 1 year post termination."
                    },
                    {
                        "clause_type": "Limitation of Liability",
                        "clause_text": "Neither party shall be liable to the other for any indirect, consequential, or punitive damages, and total liability is capped at INR 50,000.",
                        "page_reference": "Page 4, Section 11.1",
                        "risk_level": "high",
                        "risk_explanation": "The liability cap of INR 50,000 is exceptionally low and unbalanced for commercial transactions involving proprietary tech transfer.",
                        "suggested_revision": "Total liability of either party for all claims arising out of this Agreement shall be limited to the total fees paid in the 12 months preceding the claim."
                    }
                ]
            },
            "compliance": {
                "is_compliant": False,
                "summary": "Non-compliant elements detected under Section 27 of the Indian Contract Act.",
                "issues": [
                    {
                        "law_reference": "Indian Contract Act, 1872 Sec 27",
                        "clause_text": "not engage in any competing consulting services in India for a period of 3 years",
                        "violation_description": "Any agreement restraining someone from exercising a lawful profession, trade or business is void under Indian statutory law.",
                        "severity": "critical",
                        "recommended_action": "Rewrite as a narrow non-solicit clause rather than a blanket non-compete."
                    }
                ]
            },
            "draft": {
                "drafted_clause_type": "Arbitration (Mumbai Seat)",
                "drafted_text": "Any dispute arising out of this Agreement shall be referred to arbitration administered by the Mumbai Centre for International Arbitration (MCIA) in accordance with the Arbitration and Conciliation Act, 1996. The seat and venue of arbitration shall be Mumbai, and proceedings shall be conducted in English.",
                "key_terms_explained": ["MCIA administration", "Mumbai Seat & Venue", "Arbitration Act 1996"],
                "commercial_implications": "Provides a fast-track corporate dispute resolution venue under institutional rules in Mumbai."
            },
            "verification": {
                "is_grounded": True,
                "hallucinated_claims": [],
                "verified_claims": [
                    "Non-compete duration is 3 years.",
                    "Liability is capped at INR 50,000."
                ],
                "confidence_score": 1.0,
                "verification_summary": "All review assertions correspond directly to Section 8.2 and Section 11.1 of the sample document chunks."
            }
        }
    }
    
    # Also index mock chunks in local memory/ChromaDB so retrievers do not error out when user searches
    chunks = [
        {"id": "0", "text": "This Agreement shall be governed by the laws of India.", "metadata": {"page_reference": "Page 1"}},
        {"id": "1", "text": "The receiving party shall not engage in any competing consulting services in India for a period of 3 years post termination.", "metadata": {"page_reference": "Page 3"}},
        {"id": "2", "text": "Neither party shall be liable to the other for any indirect, consequential, or punitive damages, and total liability is capped at INR 50,000.", "metadata": {"page_reference": "Page 4"}}
    ]
    try:
        await dense_retriever.index_chunks(chunks, demo_doc_id)
        await asyncio.to_thread(sparse_retriever.index_chunks, chunks, demo_doc_id)
    except Exception as e:
        logger.warning(f"Failed to index demo chunks: {e}. Proceeding anyway.")
        
    return mock_results

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "model": settings.nim_model,
        "embedding_model": settings.nim_embedding_model
    }

# Serve compiled frontend assets
from fastapi.staticfiles import StaticFiles
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# Entrypoint for running standard script directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)

