import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from llm.structured import (
    ContractReviewResult, RiskLevel, ComplianceResult,
    DraftingResult, VerificationResult, SupervisorIntent
)
from agents.supervisor import handle_request
from agents.contract_review import run_contract_review
from agents.compliance import run_compliance_check
from agents.drafting import run_drafting
from agents.verification import verify_output

@pytest.mark.asyncio
@patch("agents.contract_review.chat_structured")
async def test_run_contract_review(mock_chat_structured):
    # Setup mock Pydantic response
    mock_review_result = ContractReviewResult(
        document_type="NDA",
        parties=["Party A", "Party B"],
        effective_date="2026-01-01",
        clauses=[],
        overall_risk=RiskLevel.LOW,
        executive_summary="Safe contract."
    )
    mock_chat_structured.return_value = mock_review_result

    with patch("agents.contract_review._retrieve", return_value=[]):
        res = await run_contract_review("dummy_doc_id")
        assert res.document_type == "NDA"
        assert res.overall_risk == RiskLevel.LOW
        assert res.executive_summary == "Safe contract."

@pytest.mark.asyncio
@patch("agents.compliance.chat_structured")
async def test_run_compliance_check(mock_chat_structured):
    mock_compliance = ComplianceResult(
        is_compliant=True,
        issues=[],
        summary="Compliant under Indian Contract Act."
    )
    mock_chat_structured.return_value = mock_compliance

    with patch("agents.compliance._retrieve", return_value=[]):
        res = await run_compliance_check("dummy_doc_id")
        assert res.is_compliant is True
        assert "Compliant" in res.summary

@pytest.mark.asyncio
@patch("agents.drafting.chat_structured")
async def test_run_drafting(mock_chat_structured):
    mock_draft = DraftingResult(
        drafted_clause_type="Indemnity",
        drafted_text="The supplier shall indemnify...",
        key_terms_explained=["indemnify"],
        commercial_implications="Balanced."
    )
    mock_chat_structured.return_value = mock_draft

    res = await run_drafting("Draft a mutual indemnity clause", "dummy_doc_id")
    assert res.drafted_clause_type == "Indemnity"
    assert "indemnify" in res.drafted_text

@pytest.mark.asyncio
@patch("agents.verification.chat_structured")
async def test_verify_output(mock_chat_structured):
    mock_verification = VerificationResult(
        is_grounded=True,
        hallucinated_claims=[],
        verified_claims=["Claim 1 is true"],
        confidence_score=1.0,
        verification_summary="All claims grounded."
    )
    mock_chat_structured.return_value = mock_verification

    with patch("agents.verification.dense.search", return_value=[]), \
         patch("agents.verification.sparse.search", return_value=[]):
        res = await verify_output("AI text", "doc_123", ["Claim 1 is true"])
        assert res.is_grounded is True
        assert res.confidence_score == 1.0

