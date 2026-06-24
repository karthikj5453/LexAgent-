import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from retrieval.chunker import chunk_document
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.sparse import BM25Retriever
from retrieval.dense import DenseRetriever

def test_reciprocal_rank_fusion():
    # Dense results: list of dicts with ids and scores
    dense_results = [
        {"id": "doc1_0", "text": "termination clause content", "score": 0.9, "rank": 0},
        {"id": "doc1_1", "text": "indemnity limit is $1M", "score": 0.8, "rank": 1},
    ]
    # Sparse results: list of dicts with ids and scores
    sparse_results = [
        {"id": "doc1_1", "text": "indemnity limit is $1M", "score": 0.95, "rank": 0},
        {"id": "doc1_2", "text": "governing law is Indian law", "score": 0.7, "rank": 1},
    ]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_n=2)
    assert len(fused) == 2
    # doc1_1 appears in both, so it should rank highly
    assert fused[0]["id"] == "doc1_1"

def test_sparse_retriever_indexing_and_search(tmp_path):
    persist_file = str(tmp_path / "test_bm25.pkl")
    retriever = BM25Retriever(persist_path=persist_file)

    test_chunks = [
        {"id": "0", "text": "This Agreement is governed by the laws of India.", "metadata": {}},
        {"id": "1", "text": "The liability shall not exceed the contract value.", "metadata": {}},
    ]

    retriever.index_chunks(test_chunks, "test_doc_123")
    assert len(retriever.chunks_db) == 2
    assert os.path.exists(persist_file)

    # Search query
    results = retriever.search("governed by India", k=1, doc_id="test_doc_123")
    assert len(results) == 1
    assert "India" in results[0]["text"]
    assert results[0]["metadata"]["doc_id"] == "test_doc_123"

@patch("pypdf.PdfReader")
def test_chunk_document(mock_pdf_reader):
    # Setup mock PDF pages
    mock_page_1 = MagicMock()
    mock_page_1.extract_text.return_value = "This is paragraph one.\n\nThis is paragraph two."
    
    mock_page_2 = MagicMock()
    mock_page_2.extract_text.return_value = "This is page two text."
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page_1, mock_page_2]
    mock_pdf_reader.return_value = mock_reader_instance

    with patch("os.path.exists", return_value=True):
        chunks = chunk_document("dummy_path.pdf")
        
        assert len(chunks) > 0
        assert chunks[0]["metadata"]["document_name"] == "dummy_path.pdf"
        assert "Page 1" in chunks[0]["metadata"]["page_reference"]
        # Check text is extracted
        assert "paragraph one" in chunks[0]["text"]
