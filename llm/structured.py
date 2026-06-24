from pydantic import BaseModel, Field
from typing import List, Literal
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedClause(BaseModel):
    clause_type: str = Field(description="The type of the clause, e.g., Termination, Indemnification, IP Assignment, Limitation of Liability.")
    clause_text: str = Field(description="The exact text or segment of the clause from the contract.")
    page_reference: str = Field(description="The page number or section index reference where this clause was found.")
    risk_level: RiskLevel = Field(description="The risk level associated with this clause.")
    risk_explanation: str = Field(description="Detailed explanation of why this clause is risky and its commercial implications.")
    suggested_revision: str | None = Field(default=None, description="Proposed balanced redrafted version of this clause to mitigate the risks.")

class ContractReviewResult(BaseModel):
    document_type: str = Field(description="The identified type of the document, e.g., Non-Disclosure Agreement, Service Agreement.")
    parties: List[str] = Field(description="List of parties identified in the contract.")
    effective_date: str | None = Field(default=None, description="The effective date of the contract if stated.")
    clauses: List[ExtractedClause] = Field(description="List of all key extracted clauses along with their risk assessments.")
    overall_risk: RiskLevel = Field(description="The overall risk level of the entire contract.")
    executive_summary: str = Field(description="High-level summary of the contract, its purpose, and the overall risk posture.")

class ComplianceIssue(BaseModel):
    law_reference: str = Field(description="The legal reference or section violated under Indian Law, e.g., Indian Contract Act 1872 Sec 23.")
    clause_text: str = Field(description="The text in the contract that constitutes the violation.")
    violation_description: str = Field(description="Detailed explanation of the statutory violation.")
    severity: RiskLevel = Field(description="The severity level of the compliance issue.")
    recommended_action: str = Field(description="The required action to fix the violation (e.g., delete, rewrite, seek exemption).")

class ComplianceResult(BaseModel):
    is_compliant: bool = Field(description="True if no severe compliance issues are found, False otherwise.")
    issues: List[ComplianceIssue] = Field(description="List of all identified legal compliance issues.")
    summary: str = Field(description="A concise summary of the overall compliance status under Indian law.")

class VerificationResult(BaseModel):
    is_grounded: bool = Field(description="True if all claims are fully supported by source documents, False if any claims are hallucinated.")
    hallucinated_claims: List[str] = Field(description="Claims made in the analysis that cannot be found or supported by the source document chunks.")
    verified_claims: List[str] = Field(description="Claims that were successfully matched and verified against the source text.")
    confidence_score: float = Field(description="Score from 0.0 to 1.0 indicating overall factual grounding confidence.")
    verification_summary: str = Field(description="Brief summary explaining the verification outcome and key findings.")

class SupervisorIntent(BaseModel):
    intent: Literal["review", "compliance", "draft", "query", "unknown"] = Field(description="The primary intent category classified by the supervisor.")
    sub_tasks: List[str] = Field(description="List of sub-tasks needed to resolve the user's request (e.g., review, compliance, draft, verification).")
    requires_document: bool = Field(description="Whether executing this request requires an uploaded document context.")

class DraftingResult(BaseModel):
    drafted_clause_type: str = Field(description="The type of clause drafted, e.g. Indemnification, Intellectual Property.")
    drafted_text: str = Field(description="The actual drafted legal text of the clause.")
    key_terms_explained: List[str] = Field(description="Explanations of key terms or choices made in this draft.")
    commercial_implications: str = Field(description="Implications of this clause for the business/enterprise.")

