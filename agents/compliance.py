import asyncio
from llm.client import chat_structured
from llm.prompts import COMPLIANCE_SYSTEM
from llm.structured import ComplianceResult
from retrieval.dense import DenseRetriever
from retrieval.sparse import BM25Retriever
from retrieval.fusion import reciprocal_rank_fusion
import logging

logger = logging.getLogger("lexagent.agents.compliance")

dense = DenseRetriever()
sparse = BM25Retriever()

COMPLIANCE_QUERIES = [
    "governing law jurisdiction courts in India",
    "indemnity liability limits",
    "unreasonable restriction restraint of trade Section 27 Contract Act",
    "data protection privacy IT Act",
    "arbitration dispute resolution mechanism",
]

async def _retrieve(query: str, doc_id: str) -> list[dict]:
    try:
        dense_r, sparse_r = await asyncio.gather(
            dense.search(query, k=5, doc_id=doc_id),
            asyncio.to_thread(sparse.search, query, k=5, doc_id=doc_id),
        )
        return reciprocal_rank_fusion(dense_r, sparse_r, top_n=3)
    except Exception as e:
        logger.error(f"Compliance retrieval error for query '{query}': {e}")
        return []

async def run_compliance_check(doc_id: str) -> ComplianceResult:
    logger.info(f"Running compliance check for doc_id: {doc_id}")
    
    # Retrieve relevant clauses
    tasks = [_retrieve(query, doc_id) for query in COMPLIANCE_QUERIES]
    results = await asyncio.gather(*tasks)
    
    # Deduplicate chunks
    all_chunks = []
    seen = set()
    for batch in results:
        for chunk in batch:
            if chunk["id"] not in seen:
                all_chunks.append(chunk)
                seen.add(chunk["id"])
                
    if not all_chunks:
        logger.warning(f"No compliance chunks retrieved for doc_id: {doc_id}")
        context = "[No contract excerpts retrieved for compliance checking.]"
    else:
        context_parts = []
        for i, c in enumerate(all_chunks[:15]):
            page_info = c.get("metadata", {}).get("page_reference", "Unknown Page")
            context_parts.append(f"[Chunk {i+1} - {page_info}]\n{c['text']}")
        context = "\n\n---\n\n".join(context_parts)
        
    user_prompt = f"""Review the following contract excerpts for compliance under Indian Laws (e.g., Indian Contract Act 1872, Companies Act 2013, Competition Act 2002, IT Act 2000, Specific Relief Act).

CONTRACT EXCERPTS:
{context}

Analyze if there are any clauses that violate statutory regulations or are otherwise unenforceable. Cite the relevant sections and propose remediation steps."""

    try:
        return await chat_structured(
            system=COMPLIANCE_SYSTEM,
            user=user_prompt,
            output_schema=ComplianceResult,
        )
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        return ComplianceResult(
            is_compliant=True,
            issues=[],
            summary=f"Compliance check failed due to error: {str(e)}"
        )
