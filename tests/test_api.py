from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from api.main import app
import pytest

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data

@patch("api.main.chunk_document")
@patch("api.main.dense_retriever.index_chunks", new_callable=AsyncMock)
@patch("api.main.sparse_retriever.index_chunks")
def test_upload_endpoint(mock_sparse_index, mock_dense_index, mock_chunk):
    # Mock chunking output
    mock_chunk.return_value = [
        {"id": "0", "text": "This is page 1 content", "metadata": {}}
    ]

    # Create dummy pdf file content
    pdf_content = b"%PDF-1.4 dummy pdf content"
    files = {"file": ("test.pdf", pdf_content, "application/pdf")}
    
    response = client.post("/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "doc_id" in data
    assert data["chunks_indexed"] == 1

@patch("api.main.handle_request", new_callable=AsyncMock)
def test_chat_endpoint(mock_handle_request):
    # Setup mock supervisor response
    mock_handle_request.return_value = {
        "response": "Synthesized legal review response.",
        "intent": {"intent": "review", "sub_tasks": ["review"], "requires_document": True},
        "agent_results": {"review": {"document_type": "NDA", "parties": [], "overall_risk": "low", "executive_summary": "", "clauses": []}},
        "verification_passed": True
    }

    payload = {
        "message": "Please review this document",
        "session_id": "test_session_123",
        "doc_id": "dummy_doc_456"
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "intent" in data
    assert data["verification_passed"] is True
    assert data["response"] == "Synthesized legal review response."
